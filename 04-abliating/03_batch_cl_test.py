# %%
import torch
from transformer_lens import HookedTransformer
import transformer_lens.utils
from typing import Optional, Tuple, Dict
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformer_lens import utils
from datasets import load_dataset
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))

import utils.prompts as prompt_utils
from classes.sfc_data_loader import SFCDatasetLoader
from utils.enums import SupportedDatasets
import gc
# %%


# %%

# Load a model
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
indices_to_extract = [464]
clean_dataset = {k: v[indices_to_extract] if isinstance(v, (list, torch.Tensor)) else v for k, v in clean_dataset.items()}
corrupted_dataset = {k: v[indices_to_extract] if isinstance(v, (list, torch.Tensor)) else v for k, v in corrupted_dataset.items()}


# %%
def find_original_index(sampled_prompt, full_dataset):
    """
    Find the original index by comparing the prompt with the full dataset
    """
    for i, prompt in enumerate(full_dataset['prompt']):
        if torch.equal(prompt, sampled_prompt):
            return i
    return None
# %%
def sample_dataset(n_samples=300, clean_dataset=clean_dataset, corrupted_dataset=corrupted_dataset):
    """
    Randomly sample n_samples from the dataset including answer token information.
    
    Args:
        n_samples: Number of samples to randomly select
        clean_dataset: Clean dataset
        corrupted_dataset: Corrupted dataset
    """
    # Get total dataset size
    total_size = len(clean_dataset['prompt'])
    
    # Generate random indices
    indices = torch.randperm(total_size)[:n_samples]
    
    return_values = []
    
    # Original keys
    for key in ['prompt', 'answer', 'answer_pos', 'attention_mask', 'special_token_mask']:
        if clean_dataset is not None:
            return_values.append(clean_dataset[key][indices])
        if corrupted_dataset is not None:
            return_values.append(corrupted_dataset[key][indices])
    
    return return_values

# Sample 300 random cases
test_dataset = sample_dataset(n_samples=1)
num_samples = len(test_dataset[0])

# %%

print(model.to_str_tokens(test_dataset[0][0]))
print(model.to_str_tokens(test_dataset[1][0]))
print(test_dataset[2][0])
print(test_dataset[3][0])

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


# %%

def check_model_answer_logits(model: HookedTransformer, prompt: torch.Tensor, answer: torch.Tensor, last_token_pos: torch.Tensor, k: int = 5):
    logits = model(prompt)
    answer_logits = torch.softmax(logits[0, last_token_pos, :], dim=-1)
    
    # Get top k tokens
    topk_probs, topk_indices = torch.topk(answer_logits, k)
    topk_tokens_with_probs = [(model.to_string(idx.item()), prob.item()) 
                             for idx, prob in zip(topk_indices, topk_probs)]
    
    return answer_logits[answer], topk_tokens_with_probs

index = 0
print("index: ", index)
#print(model.to_string(test_dataset[0][index]))
#print(model.to_string(test_dataset[1][index]))
print("----- Clean Prompt -----")
print("Correct prompt Answer: ", model.to_str_tokens(test_dataset[2][index]))
test_logits, topk_tokens_with_probs = check_model_answer_logits(model, test_dataset[0][index], test_dataset[2][index], test_dataset[4][index])
print("Top 5 tokens: ", topk_tokens_with_probs)
print("Correct Probability for answer token: ", test_logits)
print("----- Corrupted Prompt -----")
print("Corrupted prompt Answer: ", model.to_str_tokens(test_dataset[3][index]))
test_logits, topk_tokens_with_probs = check_model_answer_logits(model, test_dataset[1][index], test_dataset[3][index], test_dataset[5][index])
print("Top 5 tokens: ", topk_tokens_with_probs)
print("Correct Probability for answer token: ", test_logits)
original_index = find_original_index(test_dataset[0][index], clean_dataset)
print("Original Index: ", original_index)


gc.collect()
torch.cuda.empty_cache()

# Selected prompt: 3, 4

# %%
# Define a generic ablation hook that zeros out the output
def ablate_hook(value: torch.Tensor, hook: Optional) -> torch.Tensor:
    return torch.zeros_like(value)

# Define a function to run ablation analysis
def run_ablation_batch_analysis(
    model: HookedTransformer,
    input_tokens: torch.Tensor,
    tokens_to_watch: list[str],
    last_real_token_pos: torch.Tensor,
    hook_types: list[str] = ["hook_mlp_out", "attn.hook_z"],
):
    
    num_samples = len(input_tokens) #[batch_size, seq_len]
    n_hook_types = len(hook_types)
    n_layers = 1
    batch_size = 1
    n_watch_tokens = len(tokens_to_watch)
    differences_tensor = torch.zeros((num_samples // batch_size, n_layers, n_hook_types, n_watch_tokens))
    

    
    for batch_start in range(0, num_samples, batch_size):
        batch_end = batch_start + batch_size
        batch_input_tokens = input_tokens[batch_start:batch_end]

        print("batch_input_tokens.shape: ", batch_input_tokens.shape)
        # Convert tokens_to_watch to their token ids ( For all Questions it's A, B, C, D, E)
        watch_token_ids = torch.tensor([model.to_single_token(token) for token in tokens_to_watch]).repeat(batch_input_tokens.shape[0], 1)
        print("watch_token_ids.shape: ", watch_token_ids.shape)
        
        watch_token_ids = watch_token_ids.to(model.cfg.device)
    
        # Get baseline logits
        baseline_logits = model(batch_input_tokens)
        print("baseline_logits.shape: ", baseline_logits.shape)
        # last real position should be taken. !!
        # gather the last real token position
        print("last_real_token_pos.shape: ", last_real_token_pos.shape)
        
        # ->
        answer_pos_idx = einops.repeat(last_real_token_pos, 'batch -> batch 1 d_vocab',
                                    d_vocab=baseline_logits.shape[-1])
        print("answer_pos_idx.shape: ", answer_pos_idx.shape)
        answer_logits_idx = baseline_logits.gather(1, answer_pos_idx).squeeze(1) # shape [batch, d_vocab]
        print("answer_logits_idx.shape: ", answer_logits_idx.shape)
        baseline_logits = answer_logits_idx.gather(1, watch_token_ids)
                # ->
        #baseline_logits = baseline_logits[:, answer_logits_idx, watch_token_ids]  # Only get logits for watched tokens over all batches # [batch_size, n_watch_tokens]
        print("baseline_logits.shape watched tokens: ", baseline_logits.shape)
    
        results = []
    
        # Run ablations for each layer and hook type
        #n_layers = model.cfg.n_layers
        n_layers = 1
        for layer in range(n_layers):
            for hook_index, hook_type in enumerate(hook_types):
                hook_name = f"blocks.{layer}.{hook_type}"
                print("hook_name: ", hook_name)
                ablated_logits = model.run_with_hooks(
                input_tokens,
                fwd_hooks=[(hook_name, ablate_hook)]
                )   # [batch_size, seq_len, n_tokens]
                print("ablated_logits.shape: ", ablated_logits.shape)   
                
                # Get logits for watched tokens on the last token
                
                # 
                ablated_logits = ablated_logits[:, last_real_token_pos, watch_token_ids]  # [batch_size, seq_len, n_watch_tokens]
                print("ablated_logits.shape watched tokens: ", ablated_logits.shape)
                
                # Calculate differences from baseline
                diffs = baseline_logits - ablated_logits  # [batch_size, seq_len, n_watch_tokens]
                print("diffs.shape: ", diffs.shape)
                print("diffs: ", diffs)
                
                # Average all results over the batch
                diffs_averaged = diffs.mean(dim=0)  # [seq_len, n_watch_tokens]
                print("diffs.shape averaged: ", diffs_averaged.shape)
                print("diffs averaged: ", diffs_averaged)
            
                # batch_size, layer, hook_type, n_watch_tokens
                # Store the averaged differences in the tensor
                differences_tensor[batch_start // batch_size, layer, hook_index] = diffs_averaged

                print("differences_tensor.shape: ", differences_tensor.shape) # [batch_size, n_layers, n_hook_types, n_watch_tokens] -> [n_layers, n_hook_types, n_watch_tokens]
                print("differences_tensor: ", differences_tensor)

                differences_tensor = differences_tensor.mean(dim=0)
                print("differences_tensor.shape averaged: ", differences_tensor.shape)

                # result = {
                #     'layer': layer,
                #     'hook_type': hook_type,
                #     'logit_diffs': {token: diff.item() for token, diff in zip(tokens_to_watch, diffs_averaged)}
                # }
                # results.append(result)
    
    # Convert to DataFrame    
    df = pd.DataFrame(results)
    return df

# %%

# %%

def plot_ablation_results(df: pd.DataFrame, tokens_to_watch: list[str], type_text: str):
    # Get unique hook types (excluding 'none' from baseline)
    hook_types = df[df['hook_type'] != 'none']['hook_type'].unique()
    
    # Create subplots - one for each hook type
    fig = make_subplots(
        rows=len(hook_types), 
        cols=1,
        subplot_titles=[f"Impact of Ablating {hook}" for hook in hook_types],
        vertical_spacing=0.15
    )
    
    # Color scheme for tokens
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    token_colors = dict(zip(tokens_to_watch, colors[:len(tokens_to_watch)]))
    
    for idx, hook_type in enumerate(hook_types, 1):
        hook_data = df[df['hook_type'] == hook_type]
        
        for token in tokens_to_watch:
            # Extract logit differences for this token
            y_values = [row['logit_diffs'][token] for _, row in hook_data.iterrows()]
            x_values = hook_data['layer'].astype(str)
            
            fig.add_trace(
                go.Bar(
                    name=token,
                    x=x_values,
                    y=y_values,
                    marker_color=token_colors[token],
                    showlegend=(idx == 1),  # Only show legend for first subplot
                    legendgroup=token,  # Group traces by token
                ),
                row=idx, 
                col=1
            )
    
    # Update layout with more spacing
    fig.update_layout(
        height=300 * len(hook_types),
        width=1200, # Increased width
        title={
            'text': f"Impact of Layer Ablations on Token Logits<br><sub>Positive values indicate that ablation decreased the logit (made it less likely) <br> Negative values indicate that ablation increased the logit (made it more likely)</sub><br><sub>{type_text}</sub>",
            'y': 0.95,  # Moved title up
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin=dict(t=130),  # Added top margin for more space
        barmode='group',
        template='plotly_white',
        legend=dict(
            yanchor="middle",
            y=0.5,
            xanchor="right",
            x=1.15
        )
    )
    
    # Update axes
    for i in range(len(hook_types)):
        #fig.update_xaxes(title_text="Layer", row=i+1, col=1)
        fig.update_yaxes(
            title_text="Logit Difference<br>(Baseline - Ablated)",
            row=i+1, 
            col=1
        )
    
    return fig
    

# %%
# Set prompt and is the model able to do the task?


example_prompt = "What is 2 + 2?"
tokens_to_watch = ["1", "2", "3", "4", "5"]
last_real_token_pos = -1
tokenize = True
type_text = f"Clean True Prompt: {example_prompt}, Answer: 4"
#example_answer = "4"

#utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)

# %%

# Clean Prompt
print(index)
original_index = 464
example_prompt = test_dataset[0][index]
tokens_to_watch = ["A", "B", "C", "D", "E"]
last_real_token_pos = test_dataset[4][index]
type_text = f"Clean Prompt: {model.to_string(test_dataset[2][index])} - Original Index: {original_index}"
# %%
results_df = run_ablation_analysis(model, example_prompt, tokens_to_watch, last_real_token_pos=last_real_token_pos, tokenize=False)
# %%
fig = plot_ablation_results(results_df, tokens_to_watch, type_text=type_text)
fig.show()
#fig.write_html(f"ablation_results_clean_index_{original_index}.html")

# %%

# Corrupted Prompt
print(index)
example_prompt = test_dataset[1][index]
tokens_to_watch = ["A", "B", "C", "D", "E"]
last_real_token_pos = test_dataset[5][index]
type_text = f"Corrupted Prompt: {model.to_string(test_dataset[3][index])} - Right Answer: {model.to_string(test_dataset[2][index])} - Original Index: {original_index}"
# %%
results_df = run_ablation_analysis(model, example_prompt, tokens_to_watch, last_real_token_pos=last_real_token_pos, tokenize=False)
# %%
fig = plot_ablation_results(results_df, tokens_to_watch, type_text=type_text)
fig.show()
fig.write_html(f"ablation_results_corrupted_index_{original_index}.html")
# %%



# %%
