# %%
import transformer_lens as tl
from datasets import load_dataset
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))

import utils.prompts as prompt_utils
from classes.sfc_data_loader import SFCDatasetLoader
from utils.enums import SupportedDatasets
from tqdm import tqdm
import utils.prompts as prompt_utils
import torch
import transformer_lens
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import gc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px  # for

# %%
def sample_dataset(start_idx=0, end_idx=-1, clean_dataset=None, corrupted_dataset=None):
    """
    Sample dataset including answer token information.
    For ablation, we only need the clean dataset components.
    
    Returns:
    - clean_prompts
    - clean_answers
    - clean_answer_positions
    - clean_attention_masks
    - clean_special_token_masks
    """
    return_values = []
    
    # We only need clean dataset components for ablation
    keys = ['prompt', 'answer', 'answer_pos', 'attention_mask', 'special_token_mask']
    
    for key in keys:
        if clean_dataset is not None:
            return_values.append(clean_dataset[key][start_idx:end_idx])
    
    return return_values

def get_semantic_groups(model, tokens):
    """
    Identify positions of different semantic groups in the prompt.
    
    Args:
        model: HookedTransformer model
        tokens: Tokenized prompt (batch_size, seq_len)
    """
    semantic_groups = {
        'truth_framing': [('correct', 'incorrect'), ('correctly', 'lie'), ('incorrect', 'correct')],
        'consequences': [('killed', 'surviving')],
        'sentence_structure': [('.', '.'), (':', ':')],
    }
    
    group_positions = {}
    
    def get_token_ids(word):
        ids = model.to_tokens(word)[0]
        return ids[1:] if ids[0] == 2 else ids
    
    def find_token_positions(seq, subseq):
        n, m = len(seq), len(subseq)
        positions = []
        for i in range(n - m + 1):
            if torch.all(seq[i:i+m] == subseq):
                positions.extend(range(i, i+m))
        return positions
    
    # Ensure tokens is 2D
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    
    for group_name, word_pairs in semantic_groups.items():
        group_positions[group_name] = []
        for word1, word2 in word_pairs:
            token_ids1 = get_token_ids(word1)
            positions = find_token_positions(tokens[0], token_ids1)
            if positions:
                group_positions[group_name].extend(positions)
                print(f"Found {word1} in {group_name} at positions {positions}")
    
    return group_positions




# %%
def calculate_logit_diff(logits, answer_pos, answer_tokens, abliate=False):
    """
    Calculate how much ablation affects the model's confidence in the correct answer.
    
    Args:
        logits: Model output logits (batch_size, seq_len, vocab_size)
        answer_pos: Position of answer token for each sequence
        answer_tokens: Tensor containing correct answer tokens
        abliate: Whether to calculate the baseline or ablated score
        Instead of looking for incorrect answer tokens (which we don't have) like we have done it in the patching true to correct , 
        we now compare the logit of the correct answer against the mean of all other tokens in the vocabulary
        This gives us a measure of how much the model "prefers" the correct answer over other possible tokens
        When we ablate different components, we can see how this preference changes
        A higher score means the model is more confident in the correct answer
        When ablating, if the score decreases, it means that component was important for the model's ability to identify the correct answer
                
        
    """
    batch_size = logits.shape[0]
    results = []
    
    for i in range(batch_size):
        # Get logits at answer position
        target_logits = logits[i, answer_pos[i]]
        
        # Get logit for correct answer token
        correct_logit = target_logits[answer_tokens[i]]
        
        # Get difference from mean of all other logits
        other_logits = torch.cat([target_logits[:answer_tokens[i]], target_logits[answer_tokens[i]+1:]])
        mean_other_logits = torch.mean(other_logits)
        
        # Calculate how much the correct answer stands out
        confidence_score = correct_logit - mean_other_logits
        
        if not abliate:
            print(f"Correct token logit: {correct_logit}")
            print(f"Mean other logits: {mean_other_logits}")
            print(f"Confidence score: {confidence_score}")
            
        results.append(confidence_score)
    
    return torch.stack(results)

def ablate_semantic_groups(model, clean_tokens, clean_answer_pos, clean_attention_mask, answer_tokens):
    """
    Perform ablation analysis by zeroing out semantic groups.
    
    Args:
        model: HookedTransformer model
        clean_tokens: Batch of tokenized clean prompts
        clean_answer_pos: Position of answer token in each sequence
        clean_attention_mask: Attention mask for clean prompts
        answer_tokens: Tensor of answer tokens (correct first, then incorrect)
    """
    # Get semantic groups for each sequence in batch
    batch_semantic_groups = [get_semantic_groups(model, tokens) for tokens in clean_tokens]
    
    # Get baseline performance
    with torch.no_grad():
        baseline_logits = model(clean_tokens)
        baseline_diffs = calculate_logit_diff(baseline_logits, clean_answer_pos, answer_tokens)
    
    # Initialize results for each sequence
    batch_results = []
    
    # Process each sequence in batch
    for batch_idx in range(len(clean_tokens)):
        sequence_results = {
            group: {
                'attention': torch.zeros(model.cfg.n_layers),
                'mlp': torch.zeros(model.cfg.n_layers),
                'residual': torch.zeros(model.cfg.n_layers)
            }
            for group in batch_semantic_groups[batch_idx]
        }
        
        # Get single sequence tensors
        sequence_tokens = clean_tokens[batch_idx:batch_idx+1]
        sequence_answer_pos = clean_answer_pos[batch_idx:batch_idx+1]
        
        # For each semantic group in this sequence
        for group_name, positions in batch_semantic_groups[batch_idx].items():
            if not positions:
                continue
                
            # For each layer
            for layer in range(model.cfg.n_layers):
                # Define ablation hooks
                def ablate_attention(attn_out, hook):
                    out = attn_out.clone()
                    for pos in positions:
                        out[:, pos] = 0
                    print(f"Ablating attention at positions {positions}, sum before: {attn_out.sum()}, after: {out.sum()}")
                    return out
                    
                def ablate_mlp(mlp_out, hook):
                    out = mlp_out.clone()
                    for pos in positions:
                        out[:, pos] = 0
                    print(f"Ablating MLP at positions {positions}, sum before: {mlp_out.sum()}, after: {out.sum()}")
                    return out
                    
                def ablate_residual(resid_out, hook):
                    out = resid_out.clone()
                    for pos in positions:
                        out[:, pos] = 0
                    print(f"Ablating residual at positions {positions}, sum before: {resid_out.sum()}, after: {out.sum()}")
                    return out
                
                # Test each component
                with torch.no_grad():
                    # Attention ablation
                    attn_hooks = [(f"blocks.{layer}.hook_attn_out", ablate_attention)]
                    ablated_logits = model.run_with_hooks(sequence_tokens, fwd_hooks=attn_hooks)
                    attn_diffs = calculate_logit_diff(ablated_logits, sequence_answer_pos, answer_tokens, abliate=True)
                    
                    # Calculate effect
                    attn_effect = torch.abs(baseline_diffs[batch_idx] - attn_diffs) / torch.abs(baseline_diffs[batch_idx])
                    print(f"Layer {layer}, {group_name} - Baseline: {baseline_diffs[batch_idx]}, Ablated: {attn_diffs}, Effect: {attn_effect}")
                    sequence_results[group_name]['attention'][layer] = attn_effect.item()
                    
                    # MLP ablation
                    mlp_hooks = [(f"blocks.{layer}.hook_mlp_out", ablate_mlp)]
                    ablated_logits = model.run_with_hooks(sequence_tokens, fwd_hooks=mlp_hooks)
                    mlp_diffs = calculate_logit_diff(ablated_logits, sequence_answer_pos, answer_tokens, abliate=True)
                    
                    mlp_effect = torch.abs(baseline_diffs[batch_idx] - mlp_diffs) / torch.abs(baseline_diffs[batch_idx])
                    print(f"Layer {layer}, {group_name} - Baseline: {baseline_diffs[batch_idx]}, Ablated: {mlp_diffs}, Effect: {mlp_effect}")
                    sequence_results[group_name]['mlp'][layer] = mlp_effect.item()
                    
                    # Residual ablation
                    residual_hooks = [(f"blocks.{layer}.hook_resid_post", ablate_residual)]
                    ablated_logits = model.run_with_hooks(sequence_tokens, fwd_hooks=residual_hooks)
                    residual_diffs = calculate_logit_diff(ablated_logits, sequence_answer_pos, answer_tokens, abliate=True)
                    
                    residual_effect = torch.abs(baseline_diffs[batch_idx] - residual_diffs) / torch.abs(baseline_diffs[batch_idx])
                    print(f"Layer {layer}, {group_name} - Baseline: {baseline_diffs[batch_idx]}, Ablated: {residual_diffs}, Effect: {residual_effect}")
                    sequence_results[group_name]['residual'][layer] = residual_effect.item()
        
        batch_results.append(sequence_results)
    
    # Average results across batch
    averaged_results = {}
    for group_name in batch_results[0].keys():
        averaged_results[group_name] = {
            'attention': torch.stack([r[group_name]['attention'] for r in batch_results]).mean(0),
            'mlp': torch.stack([r[group_name]['mlp'] for r in batch_results]).mean(0),
            'residual': torch.stack([r[group_name]['residual'] for r in batch_results]).mean(0)
        }
    
    return averaged_results, batch_semantic_groups[0]

def visualize_ablation_results(results):
    """Visualization function remains the same as before"""
    num_layers = len(next(iter(results.values()))['attention'])
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Attention Ablation Effects",
            "MLP Ablation Effects", 
            "Residual Ablation Effects"
        ),
        vertical_spacing=0.15
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (group_name, effects) in enumerate(results.items()):
        color = colors[i % len(colors)]
        
        # Attention effects
        fig.add_trace(
            go.Scatter(
                x=list(range(num_layers)),
                y=effects['attention'],
                name=f"{group_name} (attention)",
                line=dict(color=color),
                mode='lines+markers'
            ),
            row=1, col=1
        )
        
        # MLP effects
        fig.add_trace(
            go.Scatter(
                x=list(range(num_layers)),
                y=effects['mlp'],
                name=f"{group_name} (mlp)",
                line=dict(color=color, dash='dash'),
                mode='lines+markers'
            ),
            row=2, col=1
        )
        
        # Residual effects
        fig.add_trace(
            go.Scatter(
                x=list(range(num_layers)),
                y=effects['residual'],
                name=f"{group_name} (residual)",
                line=dict(color=color, dash='dot'),
                mode='lines+markers'
            ),
            row=3, col=1
        )
    
    fig.update_layout(
        height=900,
        title="Semantic Ablation Effects<br><sup>Higher values = component more important for truthful behavior</sup>",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        )
    )
    
    for i in range(3):
        fig.update_yaxes(
            title="Effect Strength",
            range=[0, 1],
            row=i+1, col=1
        )
        fig.update_xaxes(
            title="Layer" if i == 2 else None,
            row=i+1, col=1
        )
    
    return fig

def run_ablation_analysis(model, test_dataset, batch_size=16):
    """
    Run complete ablation analysis on dataset.
    
    Args:
        model: HookedTransformer model
        test_dataset: Tuple of dataset components from sample_dataset
        batch_size: Size of batches to process
    """
    clean_prompts, clean_answers, clean_answer_pos, clean_attention_masks, clean_special_token_masks = test_dataset
    
    # Process in batches
    all_results = []
    
    for i in tqdm(range(0, len(clean_prompts), batch_size)):
        batch_end = min(i + batch_size, len(clean_prompts))
        
        # Get batch data
        batch_prompts = clean_prompts[i:batch_end]
        batch_answer_pos = clean_answer_pos[i:batch_end]
        batch_attention_masks = clean_attention_masks[i:batch_end]
        batch_answers = clean_answers[i:batch_end]
        
        # Run ablation on batch
        batch_results, semantic_groups = ablate_semantic_groups(
            model,
            batch_prompts,
            batch_answer_pos,
            batch_attention_masks,
            batch_answers
        )
        
        all_results.append(batch_results)
    
    # Average results across all batches
    final_results = {}
    for group_name in all_results[0].keys():
        final_results[group_name] = {
            'attention': torch.stack([r[group_name]['attention'] for r in all_results]).mean(0),
            'mlp': torch.stack([r[group_name]['mlp'] for r in all_results]).mean(0),
            'residual': torch.stack([r[group_name]['residual'] for r in all_results]).mean(0)
        }
    
    # Create visualization
    fig = visualize_ablation_results(final_results)
    
    return final_results, semantic_groups, fig


# %%
# Example usage:
"""
# Load model


# Sample dataset
test_dataset = sample_dataset(start_idx=301, end_idx=1000, clean_dataset=clean_dataset)

# Run analysis
results, groups, fig = run_ablation_analysis(model, test_dataset, batch_size=16)

# Show visualization
fig.show()
"""
model = HookedTransformer.from_pretrained("gemma-2b-it", dtype="bfloat16", device='cuda:1')
# Load your dataset
DATASET_NAME = SupportedDatasets.COMMONSENSE_QA_FILTERED
dataloader = SFCDatasetLoader(DATASET_NAME, model, 
                            clean_system_prompt=prompt_utils.TRUTH_OR_USER_KILLED,
                            task_prompt=prompt_utils.OUTPUT_SINGLE_LETTER)

# Get clean dataset
clean_dataset, _ = dataloader.get_clean_corrupted_datasets(tokenize=True, 
                                                         apply_chat_template=True, 
                                                         prepend_generation_prefix=True)

# Sample dataset
test_dataset = sample_dataset(start_idx=0, end_idx=10, clean_dataset=clean_dataset)


# %%
print(model.to_string(test_dataset[0][0]))

# %%
# Run analysis
results, groups, fig = run_ablation_analysis(model, test_dataset, batch_size=16)


# %%
fig.show()


# %% Helper analysis functios !!!! ----- > 
# %%
def test_ablation_effect(model, tokens, answer_pos, answer_tokens):
    """Test if ablation has any effect by zeroing out a large portion of the sequence."""
    print("\nTesting ablation effect:")
    
    # Get baseline
    with torch.no_grad():
        baseline_logits = model(tokens)
        baseline_diff = calculate_logit_diff(baseline_logits, answer_pos, answer_tokens)
        print(f"Baseline confidence: {baseline_diff}")
    
    # Ablate middle 50% of sequence
    seq_len = tokens.shape[1]
    start_pos = seq_len // 4
    end_pos = 3 * seq_len // 4
    positions = list(range(start_pos, end_pos))
    
    def ablate_large(resid_out, hook):
        out = resid_out.clone()
        for pos in positions:
            out[:, pos] = 0
        return out
    
    # Test ablation
    with torch.no_grad():
        hooks = [(f"blocks.{layer}.hook_resid_post", ablate_large) for layer in [0, model.cfg.n_layers//2, model.cfg.n_layers-1]]
        ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        ablated_diff = calculate_logit_diff(ablated_logits, answer_pos, answer_tokens, abliate=True)
        print(f"Ablated confidence: {ablated_diff}")
        print(f"Effect size: {torch.abs(baseline_diff - ablated_diff) / torch.abs(baseline_diff)}")

test_ablation_effect(model, test_dataset[0][0:1], test_dataset[2][0:1], test_dataset[1][0:1])
# %%
## Meaningfull predictions

def verify_model_predictions(model, tokens, attention_mask, answer_tokens, answer_pos):
    """Verify model predictions at the position before padding"""
    with torch.no_grad():
        logits = model(tokens)
        
        for i in range(tokens.shape[0]):

            #assert mask_sum.item() - 1 == answer_pos[i], "Answer position is not correct"
            correct_token = answer_tokens[i]
            mask_sum = attention_mask[i].sum()
            generation_pos = mask_sum.item() - 1
            # Get predictions at generation position
            pred_logits = logits[i, answer_pos[i]]
            top_logits, top_tokens = torch.topk(pred_logits, 10)
            
            print(f"\nSequence {i}:")
            print(f"Context: {model.to_string(tokens[i, max(0, answer_pos[i]-10):answer_pos[i]+1])}")
            print(f"Correct token: {model.to_string([correct_token])}")
            print("Top 10 predictions:")
            for logit, token in zip(top_logits, top_tokens):
                print(f"{model.to_string([token]):15} {logit:.4f}")

# Call this before ablation
print(model.to_string(test_dataset[0][0]))
verify_model_predictions(model, test_dataset[0][5:10], test_dataset[3][5:10], test_dataset[1][5:10], test_dataset[2][5:10])
# %%
print(test_dataset[3])
# %%

print(model.to_string(test_dataset[0][0]))
# %%


