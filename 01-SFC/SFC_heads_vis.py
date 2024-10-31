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
GPU_TO_USE = 2

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
from pathlib import Path

def get_data_path(data_folder, in_colab=COLAB):
  if in_colab:
    from google.colab import drive
    drive.mount('/content/drive')

    return Path(f'/content/drive/MyDrive/{data_folder}')
  else:
    return Path(f'./{data_folder}')

# %%
datapath = get_data_path('./data')
datapath

# %%
import sys
import os

# Add the parent directory (sfc_deception) to sys.path
sys.path.append(os.path.abspath(os.path.join('..')))

# %% [markdown]
# ## Loading the model

# %%
from sae_lens import SAE, HookedSAETransformer, ActivationsStore

USE_INSTRUCT = True
PARAMS_COUNT = 9

MODEL_NAME = f'gemma-2-{PARAMS_COUNT}b' + ('-it' if USE_INSTRUCT else '')
print(f'Using {MODEL_NAME}')

model = HookedSAETransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.bfloat16)
model

# %% [markdown]
# ## Loading the data

# %%
from classes.sfc_data_loader import SFCDatasetLoader
import utils.prompts as prompts
from utils.enums import *

# %%
DATASET_NAME = SupportedDatasets.COMMONSENSE_QA_FILTERED
NUM_SAMPLES = 10

if DATASET_NAME in [SupportedDatasets.CITIES, SupportedDatasets.FACTS, SupportedDatasets.COMPANIES]:
    dataloader = SFCDatasetLoader(DATASET_NAME, model, 
                                  clean_system_prompt=prompts.TRUTH_OR_USER_KILLED,
                                  corrupted_system_prompt=prompts.LIE_OR_USER_KILLED,
                                  task_prompt=prompts.ANSWER_TRUE_FALSE,
                                  local_dataset=True, base_folder_path=datapath,
                                  )
elif DATASET_NAME in [SupportedDatasets.COMMONSENSE_QA, SupportedDatasets.COMMONSENSE_QA_FILTERED]:
    dataloader = SFCDatasetLoader(DATASET_NAME, model, 
                                clean_system_prompt=prompts.TRUTH_OR_USER_KILLED,
                                corrupted_system_prompt=prompts.LIE_OR_USER_KILLED,
                                task_prompt=prompts.OUTPUT_SINGLE_LETTER,
                                num_samples=NUM_SAMPLES)
elif DATASET_NAME in [SupportedDatasets.VERB_AGREEMENT]:
    dataloader = SFCDatasetLoader(DATASET_NAME, model,
                                  local_dataset=True, base_folder_path=datapath)
else:
    raise ValueError(f"Dataset {DATASET_NAME.value} is not supported")

# %%
if DATASET_NAME in [SupportedDatasets.VERB_AGREEMENT]:
    clean_dataset, corrupted_dataset = dataloader.get_clean_corrupted_datasets(tokenize=True, apply_chat_template=False, prepend_generation_prefix=True)
else:
    clean_dataset, corrupted_dataset = dataloader.get_clean_corrupted_datasets(tokenize=True, apply_chat_template=True, prepend_generation_prefix=True)

# %%
CONTROL_SEQ_LEN = clean_dataset['control_sequence_length'][0].item()
N_CONTEXT = clean_dataset['prompt'].shape[1]

CONTROL_SEQ_LEN, N_CONTEXT

# %%
print('Clean dataset:')
for prompt in clean_dataset['prompt'][:3]:
  print("\nPrompt:", model.to_string(prompt), end='\n\n')

  for i, tok in enumerate(prompt):
    str_token = model.to_string(tok)
    print(f"({i-CONTROL_SEQ_LEN}, {str_token})", end=' ')
  print()

print('Corrupted dataset:')
for prompt in corrupted_dataset['prompt'][:3]:
  print("\nPrompt:", model.to_string(prompt), end='\n\n')
  
  for i, tok in enumerate(prompt):
    str_token = model.to_string(tok)
    print(f"({i-CONTROL_SEQ_LEN}, {str_token})", end=' ')
  print()

# %%
# Sanity checks

# Control sequence length must be the same for all samples in both datasets
clean_ds_control_len = clean_dataset['control_sequence_length']
corrupted_ds_control_len = corrupted_dataset['control_sequence_length']

assert torch.all(corrupted_ds_control_len == corrupted_ds_control_len[0]), "Control sequence length is not the same for all samples in the dataset"
assert torch.all(clean_ds_control_len == clean_ds_control_len[0]), "Control sequence length is not the same for all samples in the dataset"
assert clean_ds_control_len[0] == corrupted_ds_control_len[0], "Control sequence length is not the same for clean and corrupted samples in the dataset"
assert clean_dataset['answer'].max().item() < model.cfg.d_vocab, "Clean answers exceed vocab size"
assert corrupted_dataset['answer'].max().item() < model.cfg.d_vocab, "Patched answers exceed vocab size"
assert (clean_dataset['answer_pos'] < N_CONTEXT).all().item(), "Answer positions exceed logits length"
assert (corrupted_dataset['answer_pos'] < N_CONTEXT).all().item(), "Answer positions exceed logits length"

# %%
def sample_dataset(start_idx=0, end_idx=-1, clean_dataset=None, corrupted_dataset=None):
    assert clean_dataset is not None or corrupted_dataset is not None, 'At least one dataset must be provided.'
    return_values = []

    for key in ['prompt', 'answer', 'answer_pos', 'attention_mask', 'special_token_mask']:
        if clean_dataset is not None:
            return_values.append(clean_dataset[key][start_idx:end_idx])
        if corrupted_dataset is not None:
            return_values.append(corrupted_dataset[key][start_idx:end_idx])

    return return_values

# %% [markdown]
# ## Setting up head visualization

# %%
import circuitsvis as cv
from IPython.display import display

# Testing that the library works
cv.examples.hello("Taras")

# %% [markdown]
# ### Selecting data for visualizations

# %%
SAMPLE_ID = 0

clean_prompt = clean_dataset['prompt'][SAMPLE_ID]
corrupted_prompt = corrupted_dataset['prompt'][SAMPLE_ID]

clean_prompt_str = model.to_string(clean_prompt)
corrupted_prompt_str = model.to_string(corrupted_prompt)

clean_prompt_str_tokens = model.to_str_tokens(clean_prompt_str, prepend_bos=False)
corrupted_prompt_str_tokens = model.to_str_tokens(corrupted_prompt_str, prepend_bos=False)

print(f'Clean prompt: {clean_prompt_str_tokens}')
print(f'\nCorrupted prompt: {corrupted_prompt_str_tokens}')

# %%
_, clean_cache = model.run_with_cache(clean_prompt, remove_batch_dim=True)
_, corrupted_cache = model.run_with_cache(corrupted_prompt, remove_batch_dim=True)

clean_cache['pattern', 0].shape, len(clean_prompt_str_tokens)

# %% [markdown]
# ## Visualizations

# %%
ATTN_LAYERS_TO_VISUALIZE = [5, 12, 13, 17]

# %% [markdown]
# ### Clean prompt

# %%
for attn_layer in [5]:
    print(f'Inspecting layer {attn_layer} attention heads')
    attention_head_names=[f"L{attn_layer}H{i}" for i in range(12)],

    attention_pattern = clean_cache['pattern', attn_layer] 
    display(cv.attention.attention_patterns(
        tokens=clean_prompt_str_tokens, 
        attention=attention_pattern
    ))

# %%
# %% [markdown]
# 1. Layer 5: 5, 


