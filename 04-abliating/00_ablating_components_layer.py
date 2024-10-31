# %%
import torch
from transformer_lens import HookedTransformer
from typing import Optional, Tuple, Dict
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Load a model
model = HookedTransformer.from_pretrained("gpt2-small")


# %%
# Define a generic ablation hook that zeros out the output
def ablate_hook(value: torch.Tensor, hook: Optional) -> torch.Tensor:
    return torch.zeros_like(value)

# Define a function to run ablation analysis
def run_ablation_analysis(
    model: HookedTransformer,
    text: str,
    tokens_to_watch: list[str],
    hook_types: list[str] = ["hook_mlp_out", "attn.hook_z"],
) -> pd.DataFrame:
    # Convert text to input tokens
    input_tokens = model.to_tokens(text)
    
    # Convert tokens_to_watch to their token ids
    watch_token_ids = [model.to_single_token(token) for token in tokens_to_watch]
    
    # Get baseline logits
    baseline_logits = model(input_tokens)
    baseline_logits = baseline_logits[0, -1, watch_token_ids]  # Only get logits for watched tokens
    results = []
    
    
    # Run ablations for each layer and hook type
    n_layers = model.cfg.n_layers
    for layer in range(n_layers):
        for hook_type in hook_types:
            hook_name = f"blocks.{layer}.{hook_type}"
            ablated_logits = model.run_with_hooks(
                input_tokens,
                fwd_hooks=[(hook_name, ablate_hook)]
            )
            
            # Get logits for watched tokens
            ablated_logits = ablated_logits[0, -1, watch_token_ids]
            
            # Calculate differences from baseline
            diffs = baseline_logits - ablated_logits
            
            result = {
                'layer': layer,
                'hook_type': hook_type,
                'logit_diffs': {token: diff.item() for token, diff in zip(tokens_to_watch, diffs)}
            }
            results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    return df

# %%

# %%

def plot_ablation_results(df: pd.DataFrame, tokens_to_watch: list[str]):
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
    
    # Update layout
    fig.update_layout(
        height=300 * len(hook_types),
        title={
            'text': "Impact of Layer Ablations on Token Logits<br><sub>Positive values indicate that ablation decreased the logit (made it less likely) - Negative values indicate that ablation increased the logit (made it more likely)</sub>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
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
        fig.update_xaxes(title_text="Layer", row=i+1, col=1)
        fig.update_yaxes(
            title_text="Logit Difference<br>(Baseline - Ablated)",
            row=i+1, 
            col=1
        )
    
    fig.show()

# %%


text = "Lie if you can: What is 2 + 2?"
tokens_to_watch = ["2", "4", "6", "7", "10"]

results_df = run_ablation_analysis(model, text, tokens_to_watch)
plot_ablation_results(results_df, tokens_to_watch)

# %%
