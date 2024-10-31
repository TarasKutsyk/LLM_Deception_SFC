# %%
try:
    import google.colab # type: ignore
    from google.colab import output
    COLAB = True
    %pip install sae-lens transformer-lens
except:
    COLAB = False
    from IPython import get_ipython # type: ignore
    ipython = get_ipython(); assert ipython is not None
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

# Standard imports
import os
import torch
import numpy as np
from tqdm import tqdm
import plotly.express as px
import pandas as pd
import einops
from jaxtyping import Float, Int
from torch import Tensor

torch.set_grad_enabled(False)

# Device setup
GPU_TO_USE = 1

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = f"cuda:{GPU_TO_USE}" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

# utility to clear variables out of the memory & and clearing cuda cache
import gc
def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

# %%
(2 * 16 * 3000 * 16000 * 2 * 42 + 16 * 3000 * 131000 * 2 * 42) / 8589934592

# %%
from pathlib import Path

def get_data_path(data_folder, in_colab=COLAB):
  if in_colab:
    from google.colab import drive
    drive.mount('/content/drive')

    return Path(f'/content/drive/MyDrive/{data_folder}')
  else:
    return Path(f'./{data_folder}')
  
datapath = get_data_path('./data')
datapath

# %%
import sys
import os

# Add the parent directory (sfc_deception) to sys.path
sys.path.append(os.path.abspath(os.path.join('..')))

from classes.sfc_model import *

# %%
import pickle

def load_dict(filename):
    with open(filename, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict

# %% [markdown]
# ## Analysis utils

# %%
def format_component_name(component, feat_idx=None):
    """
    Formats component name for readability with predictable structure.

    Args:
        component (str): Original component name from cache.
        feat_idx (int, optional): Feature index for components with 'hook_sae_acts_post'.

    Returns:
        str: Formatted component name in the format "{layer}__{type}__{category}".
    """
    parts = component.split('.')
    layer = parts[1]  # Extract the layer number
    component_type = parts[2]  # The type (resid, mlp, or attn)
    category = "error" if 'hook_sae_error' in component else str(feat_idx) if feat_idx is not None else "unknown"

    def component_type_to_string(component_type):
        if component_type == "hook_resid_post":
            return "resid_post"
        elif component_type == "hook_mlp_out":
            return "mlp_out"
        elif component_type == "attn":
            return "attn_z"
        else:
            return "unknown"
        
    component_type = component_type_to_string(component_type)
    
    if 0 <= int(layer) <= 9:
        layer = f'0{layer}'

    return f"{layer}__{component_type}__{category}"

def get_contributing_components(cache, threshold):
    """
    Identifies components in the cache whose contribution scores are greater than a given threshold.
    Works with scalar and 1-dimensional tensors, omitting token positions in the output.

    Args:
        cache (dict): Dictionary with keys representing component names and values being tensors of contribution scores.
        threshold (float): The threshold value above which components are considered significant.

    Returns:
        tuple: A tuple containing:
            - list: List of tuples, each containing (component_name, contribution_score) sorted by contribution scores in descending order.
            - int: Total count of contributing components across all entries.
    """
    contributing_components = []
    total_count = 0

    for component, tensor in cache.items():
        if 'hook_sae_error' in component:
            if tensor.item() > threshold:
                formatted_name = format_component_name(component)
                contributing_components.append((formatted_name, tensor.item()))
                total_count += 1

        elif 'hook_sae_acts_post' in component:
            high_contrib_indices = torch.where(tensor > threshold)[0]

            for idx in high_contrib_indices:
                component_name = format_component_name(component, idx.item())
                contributing_components.append((component_name, tensor[idx].item()))
                total_count += 1

    sorted_contributing_components = sorted(contributing_components, key=lambda x: x[1], reverse=True)

    return sorted_contributing_components, total_count

# %%

def summarize_contributing_components(contributing_components, show_layers=False):
    """
    Generates hierarchical summary statistics for a flat list of contributing components,
    with optional layer-level details.

    Args:
        contributing_components (list of tuples): List of tuples where each tuple contains 
                                                  (component_name, contribution_score).
        show_layers (bool): If True, includes layer statistics in the output.

    Returns:
        dict: Hierarchical overview dictionary with summary statistics for each component type (resid, mlp, attn).
    """
    overview = {
        'resid_post': {'total': 0, 'Latents': 0, 'Errors': 0, 'layers': {} if show_layers else None},
        'mlp_out': {'total': 0, 'Latents': 0, 'Errors': 0, 'layers': {} if show_layers else None},
        'attn_z': {'total': 0, 'Latents': 0, 'Errors': 0, 'layers': {} if show_layers else None}
    }

    for component_name, score in contributing_components:
        # Extract component type and layer information
        layer, component_type, category = component_name.split('__')
        layer = int(layer)  # Convert layer to integer for numeric sorting
        is_error = category == 'error'

        # Update the main component type counters
        overview[component_type]['total'] += 1
        if is_error:
            overview[component_type]['Errors'] += 1
        else:
            overview[component_type]['Latents'] += 1

        # Update layer-specific stats if layer statistics are enabled
        if show_layers:
            if layer not in overview[component_type]['layers']:
                overview[component_type]['layers'][layer] = {'total': 0, 'Latents': 0, 'Errors': 0}
            overview[component_type]['layers'][layer]['total'] += 1
            overview[component_type]['layers'][layer]['Errors' if is_error else 'Latents'] += 1

    # Remove 'layers' key if show_layers is False
    if not show_layers:
        for component_type in overview:
            del overview[component_type]['layers']

    return overview

# %%
# ################ Version 3 ################
import numpy as np
import plotly.graph_objects as go

import numpy as np
import plotly.graph_objects as go

def create_activation_visualization_plotly(data, K=10, group_padding=0.5, layer_padding=1.5):
    """
    Creates an activation visualization using Plotly.

    Parameters:
    - data: List of tuples, where each tuple contains (node_str, score).
    - K: Percentile step size (e.g., K=10 creates groups for 0-10%, 10-20%, ..., 90-100%).
    - group_padding: Additional horizontal spacing between groups within the same (layer, hook_point).
    - layer_padding: Additional vertical spacing between layers.

    Returns:
    - fig: Plotly Figure object.
    """
    # Define hook point columns
    hook_point_columns = {'attn_z': -12, 'mlp_out': 0, 'resid_post': 12}
    base_layer_spacing = 2.0
    layer_spacing = base_layer_spacing + layer_padding  # Base vertical spacing with added padding
    node_spacing = 1.0  # Minimal horizontal spacing between nodes to prevent overlap

    # Define color scales for each hook point
    hook_point_colorscales = {
        'attn_z': 'Greens',
        'mlp_out': 'Purples',
        'resid_post': 'Blues'
    }

    # Parse data
    layer_hook_nodes = {}
    all_nodes = []

    for node_str, score in data:
        parts = node_str.split('__')
        layer = int(parts[0])
        hook_point = parts[1]
        node_id = parts[-1]
        is_error = 'error' in node_str
        node_info = {
            'id': node_str,
            'layer': layer,
            'hook_point': hook_point,
            'node_id': node_id,
            'score': score,
            'is_error': is_error
        }
        all_nodes.append(node_info)

        if layer not in layer_hook_nodes:
            layer_hook_nodes[layer] = {'attn_z': [], 'mlp_out': [], 'resid_post': []}
        layer_hook_nodes[layer][hook_point].append(node_info)

    # Function to group nodes by percentile ranges
    def group_nodes_by_percentile(nodes, K):
        # K is the percentile step size
        sorted_nodes = sorted(nodes, key=lambda x: x['score'])
        n = len(sorted_nodes)
        if n == 0:
            return []
        N = int(100 / K)
        percentiles = [i * K for i in range(N + 1)]  # e.g., for K=10, percentiles = [0,10,20,...,100]
        groups = []
        for i in range(len(percentiles) - 1):
            lower_p = percentiles[i]
            upper_p = percentiles[i + 1]
            # Calculate the indices corresponding to these percentiles
            start_idx = int(np.floor(lower_p * n / 100.0))
            end_idx = int(np.floor(upper_p * n / 100.0)) if upper_p < 100 else n
            group_nodes_in_percentile = sorted_nodes[start_idx:end_idx]
            if group_nodes_in_percentile:
                group_label = f"{lower_p}-{upper_p}%"
                group_info = {
                    'layer': group_nodes_in_percentile[0]['layer'],
                    'hook_point': group_nodes_in_percentile[0]['hook_point'],
                    'group_label': group_label,
                    'is_error': group_nodes_in_percentile[0]['is_error'],
                    'nodes': group_nodes_in_percentile,
                    'total_score': np.sum([node['score'] for node in group_nodes_in_percentile])
                }
                groups.append(group_info)
        return groups

    # Group nodes
    group_nodes = []
    for layer in layer_hook_nodes:
        for hook_point in layer_hook_nodes[layer]:
            nodes = layer_hook_nodes[layer][hook_point]
            if nodes:
                # Separate error and normal nodes
                error_nodes = [node for node in nodes if node['is_error']]
                normal_nodes = [node for node in nodes if not node['is_error']]
                # Group error nodes
                if error_nodes:
                    groups = group_nodes_by_percentile(error_nodes, K)
                    group_nodes.extend(groups)
                # Group normal nodes
                if normal_nodes:
                    groups = group_nodes_by_percentile(normal_nodes, K)
                    group_nodes.extend(groups)

    # Calculate max total_score per hook_point for normalization
    hook_point_max_scores = {}
    for hook_point in hook_point_columns.keys():
        relevant_groups = [g for g in group_nodes if g['hook_point'] == hook_point]
        if relevant_groups:
            hook_point_max_scores[hook_point] = max(g['total_score'] for g in relevant_groups)
        else:
            hook_point_max_scores[hook_point] = 1  # Prevent division by zero

    # Calculate positions with added padding
    positions = {}
    for layer in sorted(set(group['layer'] for group in group_nodes)):
        for hook_point in hook_point_columns:
            groups = [group for group in group_nodes if group['layer'] == layer and group['hook_point'] == hook_point]
            if groups:
                base_x = hook_point_columns[hook_point]
                # Total horizontal spacing: (number of groups -1) * (node_spacing + group_padding)
                total_width = (len(groups) - 1) * (node_spacing + group_padding)
                start_x = base_x - total_width / 2 if len(groups) > 1 else base_x
                for idx, group in enumerate(groups):
                    x = start_x + idx * (node_spacing + group_padding) if len(groups) > 1 else base_x
                    y = layer * layer_spacing
                    group['x'] = x
                    group['y'] = y

    # Prepare data for Plotly
    # Organize groups by node type and hook_point for separate traces
    normal_groups = {}
    error_groups = {}
    for group in group_nodes:
        if group['is_error']:
            error_groups.setdefault(group['hook_point'], []).append(group)
        else:
            normal_groups.setdefault(group['hook_point'], []).append(group)

    # Create Plotly figure
    fig = go.Figure()

    # Define symbols based on group size
    def get_symbol(is_error, group_size):
        if is_error:
            return 'triangle-up'  # All error nodes use triangle
        else:
            return 'pentagon' if group_size > 1 else 'square'  # Group nodes: pentagon; single nodes: square

    # Add normal groups
    for hook_point, groups in normal_groups.items():
        if not groups:
            continue
        max_score = hook_point_max_scores[hook_point]
        normalized_scores = [g['total_score'] / max_score for g in groups]
        fig.add_trace(go.Scatter(
            x=[g['x'] for g in groups],
            y=[g['y'] for g in groups],
            mode='markers',
            marker=dict(
                symbol=[get_symbol(False, len(g['nodes'])) for g in groups],
                size=20,
                color=normalized_scores,  # Normalized score for color mapping
                colorscale=hook_point_colorscales[hook_point],
                cmin=0,
                cmax=1,
                showscale=False,  # Remove the colorbar
                line=dict(width=1, color='Black')
            ),
            text=[
                (
                    f"Hook Point: {g['hook_point']}<br>"
                    f"Group: {g['group_label']}<br>"
                    f"Total Score: {g['total_score']:.3f}<br>"
                    f"Nodes:<br>" + "<br>".join([f"{node['id']}: {node['score']:.3f}" for node in g['nodes']])
                )
                for g in groups
            ],
            hoverinfo='text',
            showlegend=False  # Hide from legend
        ))

    # Add error groups
    for hook_point, groups in error_groups.items():
        if not groups:
            continue
        max_score = hook_point_max_scores[hook_point]
        normalized_scores = [g['total_score'] / max_score for g in groups]
        fig.add_trace(go.Scatter(
            x=[g['x'] for g in groups],
            y=[g['y'] for g in groups],
            mode='markers',
            marker=dict(
                symbol=[get_symbol(True, len(g['nodes'])) for g in groups],
                size=20,
                color=normalized_scores,  # Normalized score for color mapping
                colorscale=hook_point_colorscales[hook_point],
                cmin=0,
                cmax=1,
                showscale=False,  # Remove the colorbar
                line=dict(width=1, color='Black')
            ),
            text=[
                (
                    f"Hook Point: {g['hook_point']}<br>"
                    f"Group: {g['group_label']}<br>"
                    f"Total Score: {g['total_score']:.3f}<br>"
                    f"Nodes:<br>" + "<br>".join([f"{node['id']}: {node['score']:.3f}" for node in g['nodes']])
                )
                for g in groups
            ],
            hoverinfo='text',
            showlegend=False  # Hide from legend
        ))

    # Create custom legend
    # 1. Triangle for Error Nodes
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            symbol='triangle-up',
            size=20,
            color='gray',  # Color is irrelevant for legend
            line=dict(width=1, color='Black')
        ),
        name='Error Nodes',
        showlegend=True
    ))

    # 2. Rectangle for Latent Nodes (single nodes)
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            symbol='square',
            size=20,
            color='gray',
            line=dict(width=1, color='Black')
        ),
        name='Latent Nodes',
        showlegend=True
    ))

    # 3. Pentagon for Group Nodes (multiple nodes)
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            symbol='pentagon',
            size=20,
            color='gray',
            line=dict(width=1, color='Black')
        ),
        name='Group Nodes',
        showlegend=True
    ))

    # Calculate the minimum and maximum x-values of all nodes
    all_x_values = [group['x'] for group in group_nodes]
    min_x_value = min(all_x_values) if all_x_values else -15
    max_x_value = max(all_x_values) if all_x_values else 15

    # Adjust x-axis margins to prevent overlap
    x_margin = 2.5  # Adjust this value as needed
    x_range = [min_x_value - x_margin, max_x_value + x_margin]

    # Add vertical dotted lines for hook points
    y_values = [group['y'] for group in group_nodes] if group_nodes else [0]
    y_min = min(y_values) - layer_spacing if group_nodes else -layer_spacing
    y_max = max(y_values) + layer_spacing if group_nodes else layer_spacing
    for hook_point, x in hook_point_columns.items():
        fig.add_shape(
            type='line',
            x0=x, y0=y_min,
            x1=x, y1=y_max,
            line=dict(color='gray', width=1, dash='dot')
        )
        # Add hook point labels
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y_max + 0.5],
            mode='text',
            text=[hook_point],
            textposition="top center",
            showlegend=False
        ))

    # Add layer labels on y-axis
    unique_layers = sorted(set(group['layer'] for group in group_nodes)) if group_nodes else []
    layer_labels = [f'Layer {layer}' for layer in unique_layers]
    layer_y = [layer * layer_spacing for layer in unique_layers]

    # Position layer labels to the left of the minimum x-value
    label_x_position = min_x_value - x_margin  # Adjusted position to prevent overlap

    fig.add_trace(go.Scatter(
        x=[label_x_position] * len(layer_y),
        y=layer_y,
        mode='text',
        text=layer_labels,
        textposition="middle right",
        showlegend=False
    ))

    # Update layout
    fig.update_layout(
        title='SFC node (attr patching) scores map',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=x_range  # Set the x-axis range to include margins
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        shapes=[],  # Shapes are already added
        showlegend=True,
        width=1200,
        height=800 + len(unique_layers) * layer_padding * 50,  # Adjusted height for vertical padding
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=100, r=100, t=100, b=100)  # Adjust margins as needed
    )

    return fig

# %% [markdown]
# ## Aggregation case

# %%
aggregation_type = AttributionAggregation.ALL_TOKENS
THRESHOLD = 0.02
NODES_PREFIX = ''# 'resid_saes_128k'

def get_nodes_fname(truthful_nodes=True, nodes_prefix=NODES_PREFIX):
    nodes_type = 'truthful' if truthful_nodes else 'deceptive'
    if nodes_prefix:
        fname = f'{aggregation_type.value}_agg_{nodes_prefix}_{nodes_type}_scores.pkl'
    else:
        fname = f'{aggregation_type.value}_agg_{nodes_type}_scores.pkl'

    return datapath / fname

# %% [markdown]
# ### Truthful nodes


# %%
aggregation_type = AttributionAggregation.ALL_TOKENS

truthful_nodes_fname = get_nodes_fname(truthful_nodes=True)

truthful_nodes_scores = load_dict(truthful_nodes_fname)
truthful_nodes_scores['blocks.0.attn.hook_z.hook_sae_error'].shape

# %%
truthful_nodes_scores, total_components = get_contributing_components(truthful_nodes_scores, THRESHOLD)
print(f"Total contributing components: {total_components}")

truthful_nodes_scores

# %%
node_scores_summary = summarize_contributing_components(truthful_nodes_scores, show_layers=True)
node_scores_summary

# %%
PERCENTILE = 20
fig = create_activation_visualization_plotly(truthful_nodes_scores, K=PERCENTILE, group_padding=0.5, 
                                             layer_padding=1)

# Display the figure
fig.show()

# %%
# Save the figure as an HTML file
import plotly.io as pio

output_file = './plots/' + f'truthful_nodes_{NODES_PREFIX}_threshold_{THRESHOLD}_percentile_{PERCENTILE}.html'
pio.write_html(fig, file=output_file, auto_open=False)  # `auto_open=True` will open the HTML file in a browser

print(f"The interactive figure has been saved as {output_file}.")

# %% [markdown]
# ### Deceptive nodes

# %%
aggregation_type = AttributionAggregation.ALL_TOKENS

nodes_prefix = '' # NODES_PREFIX

deceptive_nodes_fname = get_nodes_fname(truthful_nodes=False, nodes_prefix=nodes_prefix)

deceptive_nodes_scores = load_dict(deceptive_nodes_fname)
deceptive_nodes_scores.keys()

# %%
THRESHOLD=0.02

deceptive_nodes_scores, total_components = get_contributing_components(deceptive_nodes_scores, THRESHOLD)
print(f"Total contributing components: {total_components}")

deceptive_nodes_scores

# %%
summarize_contributing_components(deceptive_nodes_scores, show_layers=True)

# %%
PERCENTILE = 20

fig = create_activation_visualization_plotly(deceptive_nodes_scores, K=PERCENTILE, group_padding=0.5, 
                                             layer_padding=1)

# Display the figure
fig.show()

# %%
output_file = './plots/' + f'deceptive_{nodes_prefix}_threshold_{THRESHOLD}_percentile_{PERCENTILE}.html'
pio.write_html(fig, file=output_file, auto_open=False)  # `auto_open=True` will open the HTML file in a browser

print(f"The interactive figure has been saved as {output_file}.")
# %%
