# %%[markdown]
# # Patching Gemma-9b-it
# Idea: Use Activation Patching to identify the important layers for lying detection.
# Todo: Double check why the semantic groups truth_related and lie_related have the same effect values in the plot. 
# Todo: Check also all the other groups, if the differ, or if there is a mistake. 

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
# 1. Load the Gemma-2b-it model
model = HookedTransformer.from_pretrained("gemma-2-9b-it", dtype="bfloat16")
# %%

DATASET_NAME = SupportedDatasets.COMMONSENSE_QA_FILTERED
dataloader = SFCDatasetLoader(DATASET_NAME, model, 
                              clean_system_prompt = prompt_utils.TRUTH_OR_USER_KILLED,
                              corrupted_system_prompt=prompt_utils.LIE_OR_USER_KILLED,
                              task_prompt=prompt_utils.OUTPUT_SINGLE_LETTER)

# %%
clean_dataset, corrupted_dataset = dataloader.get_clean_corrupted_datasets(tokenize=True, apply_chat_template=True, prepend_generation_prefix=True)


# %%
# Sample Dataset

def sample_dataset(start_idx=0, end_idx=-1, clean_dataset=clean_dataset, corrupted_dataset=corrupted_dataset):
    """
    Sample dataset including answer token information.
    """
    return_values = []
    
    # Original keys
    for key in ['prompt', 'answer', 'answer_pos', 'attention_mask', 'special_token_mask']:
        if clean_dataset is not None:
            return_values.append(clean_dataset[key][start_idx:end_idx])
        if corrupted_dataset is not None:
            return_values.append(corrupted_dataset[key][start_idx:end_idx])
    
    return return_values

test_dataset = sample_dataset(start_idx=0, end_idx=2)

# 0 = Clean Prompt
# 1 = Corrupted Prompt
# 2 = Clean Answer
# 3 = Corrupted Answer
# 4 = Clean answer_pos
# 5 = Corrupted answer_pos
# 6 = Clean attention_mask
# 7 = Corrupted attention_mask
# 8 = Clean special_token_mask
# 9 = Corrupted special_token_mask

# %%

# Test of tokenization.
print(model.to_str_tokens(test_dataset[2][0]))
print(model.to_str_tokens(test_dataset[3][0]))
print(test_dataset[2][0], test_dataset[3][0])
print(torch.cat([test_dataset[2][0].unsqueeze(0), test_dataset[3][0]]))




# %%


# use the attention mask to get the sequence to find the last position of padding by sum the mask
last_clean_token_pos = sum(test_dataset[6][0])-1
last_corrupt_token_pos = sum(test_dataset[7][0])-1

clean_prompt = test_dataset[0][0]
corrupt_prompt = test_dataset[1][0]


# %%

print(model.to_string(clean_prompt))
print(model.to_string(corrupt_prompt))

# %%
# Test tokenization
clean_tokens = clean_prompt
corrupt_tokens = corrupt_prompt


# %%
# 3. Run the model and cache activations

# %%
def get_semantic_groups(model, clean_tokens, corrupt_tokens):
    """
    Group tokens into semantic categories with proper BOS token handling.
    """
    # Ensure tokens are 2D tensors
    if clean_tokens.dim() == 1:
        clean_tokens = clean_tokens.unsqueeze(0)
    if corrupt_tokens.dim() == 1:
        corrupt_tokens = corrupt_tokens.unsqueeze(0)
        
    semantic_groups = {
        'truth_lie_related': [('incorrect', 'correct'), ('correctly', 'lie')],
        'consequences': [('killed', 'killed')],
        #'consequence': [('killed', 'surviving')],
        #'intent': [('strategy', 'strategy')],  # same word if no change
        #'question': [('question', 'question'), ('?', '?')],
        #'answer': [('answer', 'answer'), (':', ':')],
        #'user': [('user', 'user'), ('captive', 'captive')]
    }

    def get_token_ids(word):
        """Get token IDs without BOS token."""
        tokens = model.to_tokens(word)[0]
        # Remove BOS token (ID 2) if it's the first token
        if tokens[0] == 2:
            tokens = tokens[1:]
        return tokens
    
    def find_subsequence(seq, subseq):
        """Find all occurrences of subsequence in sequence."""
        n = len(seq)
        m = len(subseq)
        positions = []
        for i in range(n - m + 1):
            if torch.all(seq[i:i+m] == subseq):
                positions.extend(range(i, i+m))
        return positions
    
    # print("\nTokenization analysis:")
    # print(f"Clean tokens shape: {clean_tokens.shape}")
    # print(f"Corrupt tokens shape: {corrupt_tokens.shape}")

    token_mappings = {}  # Will store {clean_pos: corrupt_pos} mappings
    semantic_positions = {group: [] for group in semantic_groups}

    for group_name, keyword_pairs in semantic_groups.items():
        # print(f"\nProcessing group: {group_name}")
        
        for clean_word, corrupt_word in keyword_pairs:
            # Get token IDs for both versions
            clean_tokens_ids = get_token_ids(clean_word)
            corrupt_tokens_ids = get_token_ids(corrupt_word)
            
            # Find positions in both sequences
            try:
                clean_pos = find_subsequence(clean_tokens[0][1:], clean_tokens_ids)
                corrupt_pos = find_subsequence(corrupt_tokens[0][1:], corrupt_tokens_ids)
                
                # Adjust for BOS token
                clean_pos = [p + 1 for p in clean_pos]
                corrupt_pos = [p + 1 for p in corrupt_pos]
                
                if clean_pos and corrupt_pos:
                    # If we have same number of occurrences, map them in order
                    if len(clean_pos) == len(corrupt_pos):
                        for c_pos, d_pos in zip(clean_pos, corrupt_pos):
                            token_mappings[c_pos] = d_pos
                            semantic_positions[group_name].append((c_pos, d_pos))
                    #        print(f"Mapped '{clean_word}' at {c_pos} to '{corrupt_word}' at {d_pos}")
                    else:
                        print(f"Warning: Unequal occurrences of '{clean_word}' ({len(clean_pos)}) "
                              f"and '{corrupt_word}' ({len(corrupt_pos)})")
            except Exception as e:
                print(f"Error processing words '{clean_word}' and '{corrupt_word}': {str(e)}")
                continue
    
    # Validate mappings
    # if token_mappings:
    #     try:
    #         clean_tokens_str = [model.to_string(clean_tokens[0, i:i+1]) for i in token_mappings.keys()]
    #         corrupt_tokens_str = [model.to_string(corrupt_tokens[0, i:i+1]) for i in token_mappings.values()]
    #         print("\nToken mapping validation:")
    #         for (c_pos, d_pos), c_tok, d_tok in zip(token_mappings.items(), clean_tokens_str, corrupt_tokens_str):
    #             print(f"Clean token at {c_pos}: '{c_tok}' → Corrupt token at {d_pos}: '{d_tok}'")
    #     except Exception as e:
    #         print(f"Error during token mapping validation: {str(e)}")
    # else:
    #     print("\nNo token mappings found to validate")
    
    return semantic_positions
    


get_semantic_groups(model, clean_tokens, corrupt_tokens)

# %%


# %%


def get_last_real_positions(attention_masks):
    """
    Get the position of the last real token (before padding) for each sequence in batch.
    
    Args:
        attention_masks: torch.Tensor of shape (batch_size, seq_len)
    Returns:
        torch.Tensor of shape (batch_size,) containing last real positions
    """
    if attention_masks.dim() == 1:
        attention_masks = attention_masks.unsqueeze(0)
    return attention_masks.sum(dim=1) - 1


## Logit Diff Batch
def logit_diff_batch(logits, last_positions):
    """
    Calculate logit difference at the specified positions for each sequence in batch.
    
    Args:
        logits: torch.Tensor of shape (batch_size, seq_len, vocab_size)
        last_positions: torch.Tensor of shape (batch_size,) containing positions
    """
    batch_size = logits.shape[0]
    results = []
    
    for i in range(batch_size):
        target_token_logits = logits[i, last_positions[i]]
        top_values, top_indices = torch.topk(target_token_logits, k=5)
        
        # Print top tokens for debugging (only for first few in batch)
        if i < 3:  # Show first 3 examples
            print(f"\nTop 5 predictions for sequence {i} at position {last_positions[i]}")
            top_tokens = [model.to_string(torch.tensor([idx])) for idx in top_indices]
            for token, value in zip(top_tokens, top_values.tolist()):
                print(f"  {token}: {value:.4f}")
        
        results.append(top_values[0] - top_values[1])
    
    return torch.stack(results)

def mc_logit_diff_batch(logits, last_positions, answer_tokens, is_corrupted=False):
    """
    Calculate logit difference, accounting for whether we want truth or lies.
    
    Args:
        logits: torch.Tensor of shape (batch_size, seq_len, vocab_size)
        last_positions: torch.Tensor of shape (batch_size,) containing positions
        answer_tokens: tensor of shape (5,) where answer_tokens[0] is the correct answer token 
                      and answer_tokens[1:] are incorrect answer tokens
        is_corrupted: bool indicating if we're evaluating corrupted/lying case
    """
    batch_size = logits.shape[0]
    results = []
    
    assert len(answer_tokens) == 5, "Must provide exactly 5 answer tokens"
    
    # Loop through each sequence in the batch
    for i in range(batch_size):
        # Get logits for the target position (last token) of this sequence
        target_token_logits = logits[i, last_positions[i]]
        # Extract logits only for the multiple choice answer tokens (5 tokens total)
        mc_logits = target_token_logits[answer_tokens]
        
        correct_logit = mc_logits[0]  # First token is truth answer
        incorrect_logits = mc_logits[1:]  # Remaining tokens are incorrect answers
        highest_incorrect = torch.max(incorrect_logits)
        
        if is_corrupted:
            # For corrupted case: want highest_incorrect - correct
            # (positive value means model prefers lying)
            diff = highest_incorrect - correct_logit
        else:
            # For clean case: want correct - highest_incorrect
            # (positive value means model prefers truth)
            diff = correct_logit - highest_incorrect
        
        if i < 3:  # Debug printing
            print(f"\nPredictions for sequence {i} at position {last_positions[i]}")
            print(f"  Truth answer ({model.to_string(torch.tensor([answer_tokens[0]]))}): {correct_logit:.4f}")
            print(f"  Highest incorrect: {highest_incorrect:.4f}")
            print(f"  Difference ({'corrupted' if is_corrupted else 'clean'}): {diff:.4f}")
            print(f"  Model prefers: {'lying' if diff > 0 and is_corrupted else 'truth'}")
        
        results.append(diff)
    
    return torch.stack(results)



def perform_semantic_causal_patching(model, clean_tokens, corrupt_tokens, 
                                   clean_attention_mask, corrupt_attention_mask,
                                   answer_tokens,
                                   batch_size=16):
    """
    Perform causal patching with token_patching per sequence length and correct last token handling.
    """
    print("\nStarting semantic causal patching...")
    
    # Get last real positions for clean and corrupt sequences
    clean_last_positions = get_last_real_positions(clean_attention_mask)
    corrupt_last_positions = get_last_real_positions(corrupt_attention_mask)
    
    print(f"Clean last positions: {clean_last_positions.tolist()}")
    print(f"Corrupt last positions: {corrupt_last_positions.tolist()}")
    
    semantic_groups = get_semantic_groups(model, clean_tokens, corrupt_tokens)
    
    print(semantic_groups)
    
    if not any(positions for positions in semantic_groups.values()):
        print("Warning: No semantic groups found! Check token matching logic.")
        return None, semantic_groups
    
    print("\nRunning model with caching...")
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_tokens)

    print("\nCalculating mc baseline logit differences...")
    clean_logit_diffs = mc_logit_diff_batch(clean_logits, clean_last_positions, answer_tokens, is_corrupted=False)
    corrupt_logit_diffs = mc_logit_diff_batch(corrupt_logits, corrupt_last_positions, answer_tokens, is_corrupted=True)
    
    print(f"Clean logit diffs: {clean_logit_diffs}")
    print(f"Corrupt logit diffs: {corrupt_logit_diffs}")

    # Initialize results containers
    group_effects = {
        group: {
            'attn': torch.zeros((model.cfg.n_layers, len(clean_last_positions)), device='cpu'),
            'mlp': torch.zeros((model.cfg.n_layers, len(clean_last_positions)), device='cpu'),
            'residual': torch.zeros((model.cfg.n_layers, len(clean_last_positions)), device='cpu')
        }
        for group in semantic_groups
    }

    for layer in tqdm(range(model.cfg.n_layers), desc="Processing layers"):
        print(f"\nProcessing layer {layer}")
        
        for group_name, positions in semantic_groups.items():
            if not positions:
                continue
            
            print(f"\n  Processing group: {group_name} with positions: {positions}")
            
            def patch_group_attention(attn_out, hook, clean_cache, positions):
                """
                Patch attention outputs using position mappings.
                
                Args:
                    positions: List of tuples [(clean_pos, corrupt_pos), ...] indicating mapping positions
                """
                clean_attn = clean_cache[hook.name].to(attn_out.device)
                patched = attn_out.clone()
                
                # Create a mapping mask
                mask = torch.zeros_like(patched, dtype=torch.bool)
                for clean_pos, corrupt_pos in positions:
                    # When patching corrupted -> clean, we use corrupt_pos as index and clean_pos as source
                    mask[:, corrupt_pos] = True
                    patched[:, corrupt_pos] = clean_attn[:, clean_pos]
                
                return patched

            def patch_group_mlp(mlp_out, hook, clean_cache, positions):
                """
                Patch MLP outputs using position mappings.
                """
                clean_mlp = clean_cache[hook.name].to(mlp_out.device)
                patched = mlp_out.clone()
                
                # Create a mapping mask
                mask = torch.zeros_like(patched, dtype=torch.bool)
                for clean_pos, corrupt_pos in positions:
                    # When patching corrupted -> clean, we use corrupt_pos as index and clean_pos as source
                    mask[:, corrupt_pos] = True
                    patched[:, corrupt_pos] = clean_mlp[:, clean_pos]
                
                return patched

            def patch_group_residual(residual_out, hook, clean_cache, positions):
                """
                Patch residual outputs using position mappings.
                """
                clean_residual = clean_cache[hook.name].to(residual_out.device)
                patched = residual_out.clone()
                
                # Create a mapping mask
                mask = torch.zeros_like(patched, dtype=torch.bool)
                for clean_pos, corrupt_pos in positions:
                    # When patching corrupted -> clean, we use corrupt_pos as index and clean_pos as source
                    mask[:, corrupt_pos] = True
                    patched[:, corrupt_pos] = clean_residual[:, clean_pos]
                
                return patched

            # Create closure to pass positions to patch functions
            def make_patch_hook(patch_fn, clean_cache, positions):
                return lambda attn_out, hook: patch_fn(attn_out, hook, clean_cache, positions)
            
            # Patch attention
            print("    Patching attention...")
            with torch.no_grad():
                attn_hooks = [(
                    f"blocks.{layer}.hook_attn_out",
                    make_patch_hook(patch_group_attention, clean_cache, positions)
                )]
                patched_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=attn_hooks)
                attn_diffs = mc_logit_diff_batch(patched_logits, corrupt_last_positions, answer_tokens, is_corrupted=True)
                
                # Calculate and store effects for each sequence
                for i in range(len(clean_last_positions)):
                    if abs(clean_logit_diffs[i] - corrupt_logit_diffs[i]) > 1e-6:
                        effect = (attn_diffs[i] - corrupt_logit_diffs[i]) / (clean_logit_diffs[i] - corrupt_logit_diffs[i])
                        effect = max(0.0, min(1.0, effect.item()))
                    else:
                        effect = 0.5
                    group_effects[group_name]['attn'][layer, i] = effect

            # Patch MLP
            print("    Patching MLP...")
            with torch.no_grad():
                mlp_hooks = [(
                    f"blocks.{layer}.hook_mlp_out",
                    make_patch_hook(patch_group_mlp, clean_cache, positions)
                )]
                patched_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=mlp_hooks)
                mlp_diffs = mc_logit_diff_batch(patched_logits, corrupt_last_positions, answer_tokens, is_corrupted=True)
                
                # Calculate and store effects for each sequence
                for i in range(len(clean_last_positions)):
                    if abs(clean_logit_diffs[i] - corrupt_logit_diffs[i]) > 1e-6:
                        effect = (mlp_diffs[i] - corrupt_logit_diffs[i]) / (clean_logit_diffs[i] - corrupt_logit_diffs[i])
                        effect = max(0.0, min(1.0, effect.item()))
                    else:
                        effect = 0.5
                    group_effects[group_name]['mlp'][layer, i] = effect

            # Patch residual
            print("    Patching residual...")
            with torch.no_grad():
                residual_hooks = [(
                    f"blocks.{layer}.hook_resid_post",
                    make_patch_hook(patch_group_residual, clean_cache, positions)
                )]
                patched_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=residual_hooks)
                residual_diffs = mc_logit_diff_batch(patched_logits, corrupt_last_positions, answer_tokens, is_corrupted=True)
                
                # Calculate and store effects for each sequence
                for i in range(len(clean_last_positions)):
                    if abs(clean_logit_diffs[i] - corrupt_logit_diffs[i]) > 1e-6:
                        effect = (residual_diffs[i] - corrupt_logit_diffs[i]) / (clean_logit_diffs[i] - corrupt_logit_diffs[i])
                        effect = max(0.0, min(1.0, effect.item()))
                    else:
                        effect = 0.5
                    group_effects[group_name]['residual'][layer, i] = effect

            torch.cuda.empty_cache()

    # Average effects across batch
    for group_name in group_effects:
        group_effects[group_name]['attn'] = group_effects[group_name]['attn'].mean(dim=1)
        group_effects[group_name]['mlp'] = group_effects[group_name]['mlp'].mean(dim=1)
        group_effects[group_name]['residual'] = group_effects[group_name]['residual'].mean(dim=1)
    return group_effects, semantic_groups

def visualize_semantic_effects(group_effects, model):
    """
    Create visualization with error bands for semantic group causal effects.
    """
    if not group_effects:
        print("No effects to visualize!")
        return None

    num_layers = len(next(iter(group_effects.values()))['attn']['mean'])
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Attention Effects by Semantic Group<br><sup>0=no effect, 0.5=uncertain, 1=complete effect</sup>", 
            "MLP Effects by Semantic Group<br><sup>0=no effect, 0.5=uncertain, 1=complete effect</sup>", 
            "Residual Effects by Semantic Group<br><sup>0=no effect, 0.5=uncertain, 1=complete effect</sup>"
        ),
        vertical_spacing=0.15
    )

    # Define a fixed color palette
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99FFCC', '#FFB366']
    
    for i, (group_name, effects) in enumerate(group_effects.items()):
        color = colors[i % len(colors)]  # Cycle through colors if more groups than colors
        
        # Add attention trace with error bands
        attn_mean = effects['attn']['mean'].numpy()
        attn_std = effects['attn']['std'].numpy()
        
        # Create fill color with transparency
        fill_color = f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)'
        
        # Upper bound
        fig.add_trace(
            go.Scatter(
                x=list(range(num_layers)),
                y=attn_mean + attn_std,
                fill=None,
                mode='lines',
                line=dict(color=color, width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # Lower bound
        fig.add_trace(
            go.Scatter(
                x=list(range(num_layers)),
                y=attn_mean - attn_std,
                fill='tonexty',
                mode='lines',
                line=dict(color=color, width=0),
                fillcolor=fill_color,
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # Main attention line
        fig.add_trace(
            go.Scatter(
                x=list(range(num_layers)),
                y=attn_mean,
                name=f"{group_name} (attn)",
                line=dict(color=color),
                mode='lines+markers',
                hovertemplate=(
                    "Layer: %{x}<br>" +
                    "Effect: %{y:.4f} ± %{customdata[0]:.4f}<br>" +
                    "Group: " + group_name + " (attention)<br>" +
                    "<extra></extra>"
                ),
                customdata=np.column_stack([attn_std])
            ),
            row=1, col=1
        )

        # MLP effects
        mlp_mean = effects['mlp']['mean'].numpy()
        mlp_std = effects['mlp']['std'].numpy()
        
        # Upper bound
        fig.add_trace(
            go.Scatter(
                x=list(range(num_layers)),
                y=mlp_mean + mlp_std,
                fill=None,
                mode='lines',
                line=dict(color=color, width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
        
        # Lower bound
        fig.add_trace(
            go.Scatter(
                x=list(range(num_layers)),
                y=mlp_mean - mlp_std,
                fill='tonexty',
                mode='lines',
                line=dict(color=color, width=0),
                fillcolor=fill_color,
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
        
        # Main MLP line
        fig.add_trace(
            go.Scatter(
                x=list(range(num_layers)),
                y=mlp_mean,
                name=f"{group_name} (mlp)",
                line=dict(color=color, dash='dash'),
                mode='lines+markers',
                hovertemplate=(
                    "Layer: %{x}<br>" +
                    "Effect: %{y:.4f} ± %{customdata[0]:.4f}<br>" +
                    "Group: " + group_name + " (MLP)<br>" +
                    "<extra></extra>"
                ),
                customdata=np.column_stack([mlp_std])
            ),
            row=2, col=1
        )
        
        # Residual effects
        residual_mean = effects['residual']['mean'].numpy()
        residual_std = effects['residual']['std'].numpy()
        
        # Upper bound
        fig.add_trace(
            go.Scatter(
                x=list(range(num_layers)),
                y=residual_mean + residual_std,
                fill=None,
                mode='lines',
                line=dict(color=color, width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=3, col=1
        )
        
        # Lower bound
        fig.add_trace(
            go.Scatter(
                x=list(range(num_layers)),
                y=residual_mean - residual_std,
                fill='tonexty',
                mode='lines',
                line=dict(color=color, width=0),
                fillcolor=fill_color,
                showlegend=False,
                hoverinfo='skip'
            ),
            row=3, col=1
        )
        
        # Main residual line
        fig.add_trace(
            go.Scatter(
                x=list(range(num_layers)),
                y=residual_mean,
                name=f"{group_name} (residual)",
                line=dict(color=color, dash='dash'),
                mode='lines+markers',
                hovertemplate=(
                    "Layer: %{x}<br>" +
                    "Effect: %{y:.4f} ± %{customdata[0]:.4f}<br>" +
                    "Group: " + group_name + " (residual)<br>" +
                    "<extra></extra>"
                ),
                customdata=np.column_stack([residual_std])
            ),
            row=3, col=1
        )
        
        
    # Update layout
    fig.update_layout(
        height=1000,
        width=1200,
        title={
            'text': 'Causal Effects by Semantic Group<br><sup>Effect Scale: 0=no effect, 0.5=uncertain, 1=complete effect</sup>',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05,
            bordercolor="Black",
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Update axes
    for i in [1, 2, 3]:  # Updated to include all three subplots
        fig.update_xaxes(
            title_text="Layer",
            gridcolor='lightgray',
            zerolinecolor='lightgray',
            row=i, col=1
        )
        fig.update_yaxes(
            title_text="Effect Strength (0=none, 0.5=uncertain, 1=complete)",
            range=[0, 1],
            gridcolor='lightgray',
            zerolinecolor='lightgray',
            row=i, col=1
        )

    return fig

def save_and_display_results(group_effects, semantic_groups, model, save_path="patching_results.html"):
    """
    Save and display the results of the analysis.
    """
    print("\nPreparing results summary...")
    
    # Create detailed summary
    summary = []
    summary.append("=== Semantic Causal Patching Analysis Results ===\n")
    
    for group_name, positions in semantic_groups.items():
        summary.append(f"\nGroup: {group_name}")
        
        if group_name in group_effects:
            effects = group_effects[group_name]
            
            # Get top layers for attention
            attn_values = effects['attn']['mean'].numpy()
            top_attn_layers = np.argsort(attn_values)[-3:][::-1]
            
            # Get top layers for MLP
            mlp_values = effects['mlp']['mean'].numpy()
            top_mlp_layers = np.argsort(mlp_values)[-3:][::-1]
            
            # Get top layers for residual
            residual_values = effects['residual']['mean'].numpy()
            top_residual_layers = np.argsort(residual_values)[-3:][::-1]
            
            summary.append("  Top Attention Layers:")
            for layer in top_attn_layers:
                summary.append(f"    Layer {layer}: {attn_values[layer]:.4f}")
                
            summary.append("  Top MLP Layers:")
            for layer in top_mlp_layers:
                summary.append(f"    Layer {layer}: {mlp_values[layer]:.4f}")
                
            summary.append("  Top Residual Layers:")
            for layer in top_residual_layers:
                summary.append(f"    Layer {layer}: {residual_values[layer]:.4f}")
                
            summary.append(f"  Average Attention Effect: {attn_values.mean():.4f}")
            summary.append(f"  Average MLP Effect: {mlp_values.mean():.4f}")
            summary.append(f"  Average Residual Effect: {residual_values.mean():.4f}")
        else:
            summary.append("  No effects calculated for this group")
    
    # Save summary to file
    with open("patching_summary.txt", "w") as f:
        f.write("\n".join(summary))
    
    # Create and save visualization
    fig = visualize_semantic_effects(group_effects, model)
    if fig is not None:
        fig.write_html(save_path)
        print(f"\nResults saved to {save_path} and patching_summary.txt")
        return fig
    else:
        print("\nNo visualization created due to missing effects")
        return None

# Update the analyze_semantic_effects function to use the new visualization
def analyze_semantic_effects(model, test_dataset):
    """
    Main analysis function updated for multiple batches.
    """
    print("Starting analysis...")
    
    num_batches = len(test_dataset[0])  # Get total number of batches
    all_group_effects = []
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        # Extract data for the current batch
        clean_tokens = test_dataset[0][batch_idx]
        corrupt_tokens = test_dataset[1][batch_idx]
        clean_attention_mask = test_dataset[6][batch_idx]
        corrupt_attention_mask = test_dataset[7][batch_idx]
        answer_tokens = torch.cat([test_dataset[2][batch_idx].unsqueeze(0), test_dataset[3][batch_idx]])
        
        # Ensure tokens and attention masks are 2D tensors
        if clean_tokens.dim() == 1:
            clean_tokens = clean_tokens.unsqueeze(0)
        if corrupt_tokens.dim() == 1:
            corrupt_tokens = corrupt_tokens.unsqueeze(0)
        if clean_attention_mask.dim() == 1:
            clean_attention_mask = clean_attention_mask.unsqueeze(0)
        if corrupt_attention_mask.dim() == 1:
            corrupt_attention_mask = corrupt_attention_mask.unsqueeze(0)
        
        # Perform semantic causal patching for this batch
        batch_effects, semantic_groups = perform_semantic_causal_patching(
            model, clean_tokens, corrupt_tokens,
            clean_attention_mask, corrupt_attention_mask,
            answer_tokens
        )
        
        if batch_effects is not None:
            all_group_effects.append(batch_effects)
    
    # Aggregate results across batches
    aggregated_effects = {}
    for group_name in all_group_effects[0].keys():
        aggregated_effects[group_name] = {
            'attn': {
                'mean': torch.stack([batch[group_name]['attn'] for batch in all_group_effects]).mean(dim=0),
                'std': torch.stack([batch[group_name]['attn'] for batch in all_group_effects]).std(dim=0)
            },
            'mlp': {
                'mean': torch.stack([batch[group_name]['mlp'] for batch in all_group_effects]).mean(dim=0),
                'std': torch.stack([batch[group_name]['mlp'] for batch in all_group_effects]).std(dim=0)
            },
            'residual': {
                'mean': torch.stack([batch[group_name]['residual'] for batch in all_group_effects]).mean(dim=0),
                'std': torch.stack([batch[group_name]['residual'] for batch in all_group_effects]).std(dim=0)
            }
        }
    
    return aggregated_effects, semantic_groups

# %%
group_effects, semantic_groups = analyze_semantic_effects(model, test_dataset)
# %%
# Create visualization and save results
print("\nCreating visualization and saving results...")
fig = save_and_display_results(group_effects, semantic_groups, model)


# %%

fig.show()










# %%
