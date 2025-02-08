from transformer_lens import ActivationCache
from sae_lens import SAE, HookedSAETransformer
import torch
import numpy as np
from tqdm import tqdm
import einops
from jaxtyping import Float, Int
from torch import Tensor
from enum import Enum

torch.set_grad_enabled(False)

# utility to clear variables out of the memory & and clearing cuda cache
import gc
def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

### Utility enums ###
class NodeScoreType(Enum):
    ATTRIBUTION_PATCHING = 'ATP'
    INTEGRATED_GRADIENTS = 'INT_GRADS'

class AttributionPatching(Enum):
    NORMAL = 'TRADITIONAL' # Approximates the effect of patching activations from patched run to the clean run
    ZERO_ABLATION = 'ZERO_ABLATION' # Approximates the effect of zeroing out activations from the patched/clean run

class AttributionAggregation(Enum):
    ALL_TOKENS = 'All_tokens'
    NONE= 'None'

### Utility functions ###
def sample_dataset(start_idx=0, end_idx=-1, clean_dataset=None, corrupted_dataset=None):
    assert clean_dataset is not None or corrupted_dataset is not None, 'At least one dataset must be provided.'
    return_values = []

    for key in ['prompt', 'answer', 'answer_pos', 'attention_mask']:
        if clean_dataset is not None:
            return_values.append(clean_dataset[key][start_idx:end_idx])
        if corrupted_dataset is not None:
            return_values.append(corrupted_dataset[key][start_idx:end_idx])

    return return_values

### Main class ###
class SFC_Gemma():
    def __init__(self, model, attach_saes=True, params_count=9, control_seq_len=1, caching_device=None,
                sae_resid_release=None, sae_attn_release=None, sae_mlp_release=None,
                sae_attn_width='16k', sae_mlp_width='16k', first_16k_resid_layers=None):
        if sae_resid_release is None:
            sae_resid_release = f'gemma-scope-{params_count}b-pt-res-canonical'

        if sae_attn_release is None:
            sae_attn_release = f'gemma-scope-{params_count}b-pt-att-canonical'

        if sae_mlp_release is None:
            sae_mlp_release = f'gemma-scope-{params_count}b-pt-mlp-canonical'

        self.model = model
        self.cfg = model.cfg
        self.device = model.cfg.device
        self.caching_device = caching_device
        self.control_seq_len = control_seq_len

        if params_count == 9:
            self.model.set_use_attn_in(True)
        self.model.set_use_attn_result(True)
        self.model.set_use_hook_mlp_in(True)
        self.model.set_use_split_qkv_input(True)

        self.n_layers = self.cfg.n_layers
        self.d_model = self.cfg.d_model

        width_to_d_sae = lambda width: {
            '131k': 131072,
            '16k': 16384
        }[width]

        self.attn_d_sae = width_to_d_sae(sae_attn_width)
        self.mlp_d_sae = width_to_d_sae(sae_mlp_width)

        if first_16k_resid_layers is None:
            first_16k_resid_layers = self.n_layers

        print(f'Using 16K SAEs for the first {first_16k_resid_layers} layers, the rest {self.n_layers - first_16k_resid_layers} layer(s) - 131k SAEs')
        resid_saes_widths = ['16k'] * first_16k_resid_layers + ['131k'] * (self.n_layers - first_16k_resid_layers)

        self.resid_d_sae = [width_to_d_sae(width) for width in resid_saes_widths]

        # Initialize dictionary to store SAEs by type: resid, attn, mlp
        self.saes_dict = {
            'resid': [],
            'attn': [],
            'mlp': []
        }

        # Load all SAEs into the dictionary
        self.saes_dict['resid'] = [
            self._load_sae(sae_resid_release, f'layer_{i}/width_{resid_saes_widths[i]}/canonical') for i in range(self.n_layers)
        ]
        self.saes_dict['attn'] = [
            self._load_sae(sae_attn_release, f'layer_{i}/width_{sae_attn_width}/canonical') for i in range(self.n_layers)
        ]
        self.saes_dict['mlp'] = [
            self._load_sae(sae_mlp_release, f'layer_{i}/width_{sae_mlp_width}/canonical') for i in range(self.n_layers)
        ]
        self.saes = self.saes_dict['resid'] + self.saes_dict['mlp'] + self.saes_dict['attn']
        
        # Attach all SAEs
        if attach_saes:
            self.add_saes()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compute_node_scores_for_normal_patching(self, clean_dataset, patched_dataset, batch_size=50, total_batches=None,
                                                score_type: NodeScoreType = NodeScoreType.ATTRIBUTION_PATCHING, run_without_saes=True, N_IG = 10):
        if run_without_saes:
            print('Running without SAEs, gradients and activations will be computed analytically.')

        n_prompts, seq_len = clean_dataset['prompt'].shape
        assert n_prompts == clean_dataset['answer'].shape[0] == patched_dataset['answer'].shape[0]

        prompts_to_process = n_prompts if total_batches is None else batch_size * total_batches

        if total_batches is None:
            total_batches = n_prompts // batch_size

            if n_prompts % batch_size != 0:
                total_batches += 1

        metrics_clean_scores = []
        metrics_patched = []

        for i in tqdm(range(0, prompts_to_process, batch_size)):
            clean_prompts, corrupted_prompts, clean_answers, corrupted_answers, clean_answers_pos, corrupted_answers_pos, \
            clean_attn_mask, corrupted_attn_mask = sample_dataset(i, i + batch_size, clean_dataset, patched_dataset)

            metric_clean = lambda logits: self.get_logit_diff(logits, clean_answers, corrupted_answers, clean_answers_pos).mean()
            metric_patched = lambda logits: self.get_logit_diff(logits, clean_answers, corrupted_answers, corrupted_answers_pos).mean()

            if score_type == NodeScoreType.ATTRIBUTION_PATCHING:
                metric_clean, cache_clean, grad_clean = self.run_with_cache(clean_prompts, clean_attn_mask, metric_clean, 
                                                                            run_without_saes=run_without_saes)

                metric_patched, cache_patched, _ = self.run_with_cache(corrupted_prompts, corrupted_attn_mask, metric_patched, 
                                                                       run_backward_pass=False, run_without_saes=run_without_saes)
                # print('Fwd clean keys:', cache_clean.keys())
                # print('Grad clean keys:', grad_clean.keys())
                # print('Fwd patched keys:', cache_patched.keys())

                if i == 0:
                    node_scores = self.initialize_node_scores(cache_clean, run_without_saes=run_without_saes)
                self.update_node_scores(node_scores, grad_clean, cache_clean, total_batches, 
                                        cache_patched=cache_patched, attr_type=AttributionPatching.NORMAL, run_without_saes=run_without_saes)

                del grad_clean
            elif score_type == NodeScoreType.INTEGRATED_GRADIENTS:
                raise NotImplementedError('Integrated gradients are not implemented yet.')

            del cache_clean, cache_patched
            clear_cache()

            metrics_clean_scores.append(metric_clean)
            metrics_patched.append(metric_patched)

        clean_metric = torch.tensor(metrics_clean_scores).mean().item()
        patched_metric = torch.tensor(metrics_patched).mean().item()

        return (
            clean_metric, patched_metric,
            node_scores
        )
    
    def compute_truthful_node_scores(self, clean_dataset, corrupted_dataset, batch_size=50, total_batches=None,
                                     run_without_saes=True, aggregation_type=AttributionAggregation.ALL_TOKENS):
        if run_without_saes:
            print('Running without SAEs, gradients and activations will be computed analytically.')

        n_prompts, seq_len = clean_dataset['prompt'].shape
        assert n_prompts == clean_dataset['answer'].shape[0]

        prompts_to_process = n_prompts if total_batches is None else batch_size * total_batches
        if total_batches is None:
            total_batches = n_prompts // batch_size

            if n_prompts % batch_size != 0:
                total_batches += 1

        metrics_clean_scores = []

        for i in tqdm(range(0, prompts_to_process, batch_size)):
            clean_prompts, corrupted_prompts, clean_answers, corrupted_answers, clean_answers_pos, corrupted_answers_pos, \
                clean_attn_mask, corrupted_attn_mask = sample_dataset(i, i + batch_size, clean_dataset, corrupted_dataset)

            # Take the negative log-prob of the answer as our metric
            # metric = lambda logits: -self.get_answer_logit(logits, clean_answers, clean_answers_pos).mean()
            # Or use the standard patching metric (log dif of incorrect - correct answers)
            metric = lambda logits: self.get_logit_diff(logits, clean_answers, corrupted_answers, clean_answers_pos).mean()

            metric_clean_score, cache_clean, grad_clean = self.run_with_cache(clean_prompts, clean_attn_mask, metric,
                                                                              run_without_saes=run_without_saes)

            if i == 0:
                node_scores = self.initialize_node_scores(cache_clean, run_without_saes=run_without_saes)
            self.update_node_scores(node_scores, grad_clean, cache_clean, total_batches, run_without_saes=run_without_saes,
                                    attr_type=AttributionPatching.ZERO_ABLATION)

            del cache_clean, grad_clean
            clear_cache()

            metrics_clean_scores.append(metric_clean_score)

        clean_metric_score = torch.tensor(metrics_clean_scores).mean().item()
        self.aggregate_node_scores(node_scores, aggregation_type=aggregation_type)

        return (
            clean_metric_score,
            node_scores
        )
    
    def compute_deceptive_node_scores(self, clean_dataset, corrupted_dataset, batch_size=50, total_batches=None,
                                      run_without_saes=True, aggregation_type=AttributionAggregation.ALL_TOKENS,
                                      metric='logit_diff'):
        if run_without_saes:
            print('Running without SAEs, gradients and activations will be computed analytically.')

        n_prompts, seq_len = clean_dataset['prompt'].shape
        assert n_prompts == clean_dataset['answer'].shape[0]

        prompts_to_process = n_prompts if total_batches is None else batch_size * total_batches
        if total_batches is None:
            total_batches = n_prompts // batch_size

            if n_prompts % batch_size != 0:
                total_batches += 1

        metrics_corrupted_scores = []

        for i in tqdm(range(0, prompts_to_process, batch_size)):
            clean_prompts, corrupted_prompts, clean_answers, corrupted_answers, clean_answers_pos, corrupted_answers_pos, \
                clean_attn_mask, corrupted_attn_mask = sample_dataset(i, i + batch_size, clean_dataset, corrupted_dataset)
            
            if metric == 'logit_diff':
                # Or use the standard patching metric (log dif of correct - incorrect answers, so flipping the sign below) 
                metric = lambda logits: -self.get_logit_diff(logits, clean_answers, corrupted_answers, corrupted_answers_pos).mean()
            else:
                # Take the log-prob of the answer as our metric
                metric = lambda logits: self.get_answer_logit(logits, clean_answers, clean_answers_pos).mean()

            metric_corrupted_score, cache_corrupted, grad_corrupted = self.run_with_cache(corrupted_prompts, corrupted_attn_mask, metric,
                                                                                          run_without_saes=run_without_saes)

            if i == 0:
                node_scores = self.initialize_node_scores(cache_corrupted, run_without_saes=run_without_saes)
            self.update_node_scores(node_scores, grad_corrupted, cache_corrupted, total_batches, run_without_saes=run_without_saes,
                                    attr_type=AttributionPatching.ZERO_ABLATION)

            del cache_corrupted, grad_corrupted
            clear_cache()

            metrics_corrupted_scores.append(metric_corrupted_score)

        corrupted_metric_score = torch.tensor(metrics_corrupted_scores).mean().item()
        self.aggregate_node_scores(node_scores, aggregation_type=aggregation_type)

        return (
            corrupted_metric_score,
            node_scores
        )

    def get_component_cache(self, clean_dataset, patched_dataset, batch_size=50, total_batches=None, component='attn'):
        n_prompts, seq_len = clean_dataset['prompt'].shape
        assert n_prompts == clean_dataset['answer'].shape[0] == patched_dataset['answer'].shape[0]

        prompts_to_process = n_prompts if total_batches is None else batch_size * total_batches

        if total_batches is None:
            total_batches = n_prompts // batch_size

            if n_prompts % batch_size != 0:
                total_batches += 1

        metrics_clean_scores = []
        metrics_patched = []

        fwd_cache_filter = lambda name: f'{component}_out' in name
        bwd_cache_filter = lambda name: f'{component}_out' in name

        def init_attn_scores(cache):
            scores_dict = {}

            for key, cache_tensor in cache.items():
                batch, pos, _ = cache_tensor.shape
                scores_dict[key] =  {
                                        f'{component}_contribution': torch.zeros((pos), dtype=torch.bfloat16, device=self.caching_device),
                                        f'{component}_attribution': torch.zeros((pos), dtype=torch.bfloat16, device=self.caching_device),
                                        f'{component}_attr_patching': torch.zeros((pos), dtype=torch.bfloat16, device=self.caching_device),
                                    }

            return scores_dict
        
        def update_scores(attn_scores, cache_clean, grad_clean, cache_patched):
            assert cache_clean.keys() == grad_clean.keys() == cache_patched.keys()
            
            # We'll compute the following scores for each of the attention layers (i.e. for each key in the cache):
            # - attn_contribution = norm of the attn_out vector
            # - attn_attribution = norm of the gradient of the attn_out vector
            # - attn_attr_patching = attention gradient multiplied by the difference of activations (as in AtP/IG)
            for key in cache_clean.keys():
                # cache_clean, grad_clean, cache_patched is of shape [batch, pos, d_model] now
                attn_contribution = cache_clean[key].norm(dim=-1)
                attn_attribution = grad_clean[key].norm(dim=-1)
                attn_attr_patching = einops.einsum(grad_clean[key], cache_patched[key] - cache_clean[key],
                                                'batch pos d_model, batch pos d_model -> batch pos')

                # Reduce the scores to compute the mean across batches
                attn_contribution = attn_contribution.mean(0)
                attn_attribution = attn_attribution.mean(0)
                attn_attr_patching = attn_attr_patching.mean(0)

                # Accumulate the scores
                attn_scores[key][f'{component}_contribution'] += attn_contribution / total_batches
                attn_scores[key][f'{component}_attribution'] += attn_attribution / total_batches
                attn_scores[key][f'{component}_attr_patching'] += attn_attr_patching / total_batches

        for i in tqdm(range(0, prompts_to_process, batch_size)):
            clean_prompts, corrupted_prompts, clean_answers, corrupted_answers, clean_answers_pos, corrupted_answers_pos, \
            clean_attn_mask, corrupted_attn_mask = sample_dataset(i, i + batch_size, clean_dataset, patched_dataset)

            metric_clean = lambda logits: self.get_logit_diff(logits, clean_answers, corrupted_answers, clean_answers_pos).mean()
            metric_patched = lambda logits: self.get_logit_diff(logits, clean_answers, corrupted_answers, corrupted_answers_pos).mean()

            metric_clean, cache_clean, grad_clean = self.run_with_cache(clean_prompts, clean_attn_mask, metric_clean, 
                                                                        fwd_cache_filter=fwd_cache_filter, bwd_cache_filter=bwd_cache_filter,
                                                                        run_without_saes=True)

            metric_patched, cache_patched, _ = self.run_with_cache(corrupted_prompts, corrupted_attn_mask, metric_patched, 
                                                                   fwd_cache_filter=fwd_cache_filter, run_backward_pass=False, 
                                                                   run_without_saes=True)

            metrics_clean_scores.append(metric_clean)
            metrics_patched.append(metric_patched)

            if i == 0:
                attn_scores = init_attn_scores(cache_clean)

            update_scores(attn_scores, cache_clean, grad_clean, cache_patched)
                
            del cache_clean, cache_patched, grad_clean
            clear_cache()

        return attn_scores

    def run_with_cache(self, tokens: Int[Tensor, "batch pos"],
                       attn_mask: Int[Tensor, "batch pos"], metric,
                       fwd_cache_filter=None, bwd_cache_filter=None, run_backward_pass=True, run_without_saes=False):
        if run_without_saes:
            if self.are_saes_attached():
                raise ValueError('Forward & Backward passes are performed analytically, but SAEs are still attached. Call reset_saes() first to save VRAM.')
            if self.caching_device is None:
                raise ValueError('Caching device must be provided for no-SAEs runs.')
        
        if fwd_cache_filter is None:
            if run_without_saes:
                fwd_cache_filter = lambda name: 'resid_post' in name or 'attn.hook_z' in name or 'mlp_out' in name
            else:
                # Take the SAE latents and error term activations by default
                fwd_cache_filter = lambda name: 'hook_sae_acts_post' in name or 'hook_sae_error' in name

        cache = {}
        def forward_cache_hook(act, hook):
            cache[hook.name] = act.detach().to(self.caching_device)

        self.model.add_hook(fwd_cache_filter, forward_cache_hook, "fwd")

        grad_cache = {}

        try:
            if run_backward_pass:
                self._set_backward_hooks(grad_cache, bwd_cache_filter, compute_grad_analytically=run_without_saes)

                # Enable gradients only during the backward pass
                with torch.set_grad_enabled(True):
                    metric_value = metric(self.model(tokens, attention_mask=attn_mask))
                    metric_value.backward()  # Compute gradients
            else:
                # Forward pass only
                with torch.set_grad_enabled(False):
                    metric_value = metric(self.model(tokens, attention_mask=attn_mask))
        finally:
            clear_cache()
            # Ensure hooks are reset regardless of exceptions
            self.model.reset_hooks()
            if self.are_saes_attached():
                self._reset_sae_hooks()

        return (
            metric_value.item(),
            ActivationCache(cache, self.model).to(self.caching_device),
            ActivationCache(grad_cache, self.model).to(self.caching_device),
        )

    def _set_backward_hooks(self, grad_cache, bwd_hook_filter=None, compute_grad_analytically=False):
        if bwd_hook_filter is None:
            if compute_grad_analytically:
                bwd_hook_filter = lambda name: 'resid_post' in name or 'attn.hook_z' in name or 'mlp_out' in name
            else:
                bwd_hook_filter = lambda name: 'hook_sae_acts_post' in name or 'hook_sae_output' in name or 'hook_sae_input' in name

        temp_cache = {}

        if compute_grad_analytically:
            # Just store the gradients in the analytical case, we'll use them later
            def backward_cache_hook(gradient, hook):
                grad_cache[hook.name] = gradient.detach().to(self.caching_device)
        else:
            # Computing grads non-analytically using Pytorch autograd
            def backward_cache_hook(gradient, hook):
                if 'hook_sae_output' in hook.name:
                    hook_sae_error_name = hook.name.replace('hook_sae_output', 'hook_sae_error')
                    grad_cache[hook_sae_error_name] = gradient.detach()

                    # We're storing the gradients for the SAE output activations to copy them to the SAE input activations gradients
                    if not 'hook_z' in hook.name:
                        temp_cache[hook.name] = gradient.detach()
                    else: # In the case of attention hook_z hooks, reshape them to match the SAE input shape, which doesn't include n_heads
                        hook_z_grad = einops.rearrange(gradient.detach(),
                                                    'batch pos n_head d_head -> batch pos (n_head d_head)')
                        temp_cache[hook.name] = hook_z_grad
                elif 'hook_sae_input' in hook.name:
                    # We're copying the gradients from the SAE output activations to the SAE input activations gradients
                    sae_output_grad_name = hook.name.replace('hook_sae_input', 'hook_sae_output')

                    grad_cache[hook.name] = temp_cache[sae_output_grad_name]
                    gradient = grad_cache[hook.name]

                    # Pass-through: use the downstream gradients
                    return (gradient,)
                else:
                    # Default case (SAE latents): just store the gradients
                    grad_cache[hook.name] = gradient.detach()

        self.model.add_hook(bwd_hook_filter, backward_cache_hook, "bwd")

    def initialize_node_scores(self, cache: ActivationCache, score_type: NodeScoreType = NodeScoreType.ATTRIBUTION_PATCHING,
                              run_without_saes=True):
        node_scores = {}
        
        for key, cache_tensor in cache.items():
            # A node is either an SAE latent or an SAE error terms
            # Here it's represented as the hook-point name - cache key

            if run_without_saes:
                d_sae = self.key_to_d_sae(key)

                # cache is of shape [batch pos d_act] if not an attn hook, [batch pos n_head d_head] otherwise
                if 'hook_z' not in key:
                    batch, pos, d_act = cache_tensor.shape
                else:
                    batch, pos, n_head, d_act = cache_tensor.shape

                sae_latent_name, sae_error_name = self.hook_name_to_sae_act_name(key)

                node_scores[sae_error_name] = torch.zeros((pos), dtype=torch.bfloat16, device=self.caching_device)
                node_scores[sae_latent_name] = torch.zeros((pos, d_sae), dtype=torch.bfloat16, device=self.caching_device)

                # if '.0.' in key:
                #     print(f'Initialized scores for {key} with {sae_error_name} of shape {node_scores[sae_error_name].shape} and {sae_latent_name} of shape {node_scores[sae_latent_name].shape}.')
            else:
                if 'hook_z.hook_sae_error' not in key:
                    batch, pos, d_act = cache_tensor.shape
                else:
                    batch, pos, n_head, d_act = cache_tensor.shape

                if 'hook_sae_error' in key:
                    # An "importance value" for the SAE error is scalar - it's a single node
                    node_scores[key] = torch.zeros((pos), dtype=torch.bfloat16, device=self.device)
                else:
                    # An "importance value" for SAE latents is a vector with length d_sae (d_act)
                    node_scores[key] = torch.zeros((pos, d_act), dtype=torch.bfloat16, device=self.device)

        return node_scores
    
    def aggregate_node_scores(self, node_scores, aggregation_type):
        # print(f'Aggregating scores with type {aggregation_type}.')

        for key, score_tensor in node_scores.items():
            score_tensor_filtered = score_tensor[self.control_seq_len:]
            if aggregation_type.value == AttributionAggregation.ALL_TOKENS.value:
                # print(f'Aggregating scores for {key} with shape {score_tensor_filtered.shape}.')
                score_tensor_aggregated = score_tensor_filtered.sum(0)
                # print(f'Aggregated scores for {key} with shape {score_tensor_aggregated.shape}.')
            else:
                score_tensor_aggregated = score_tensor_filtered

            node_scores[key] = score_tensor_aggregated
    
    def update_node_scores(self, node_scores: ActivationCache, grad_cache: ActivationCache, cache_clean: ActivationCache, total_batches,
                           cache_patched=None, attr_type=AttributionPatching.NORMAL, run_without_saes=True, batch_reduce='mean'):
        if attr_type.value == AttributionPatching.NORMAL.value:
            assert cache_patched is not None, 'Patched cache must be provided for normal attribution patching.'
            
        if not run_without_saes:
            self.update_node_scores_saes_run(node_scores, grad_cache, cache_clean, total_batches, cache_patched=cache_patched,
                                             attr_type=attr_type, batch_reduce=batch_reduce)
        else:
            self.update_node_scores_no_saes_run(node_scores, grad_cache, cache_clean, total_batches, cache_patched=cache_patched,
                                                attr_type=attr_type, batch_reduce=batch_reduce)
            
    def update_node_scores_no_saes_run(self, node_scores: ActivationCache, clean_grad, cache_clean, total_batches,
                                       cache_patched=None, attr_type=AttributionPatching.NORMAL, batch_reduce='mean'):
        def compute_score_update(key):
            """
            Computes the attribution patching score to add for the current batch and key.
            A key is either 
                - run_without_saes=True: an SAE latent or an SAE error term.
                - run_without_saes=False: a hook-point name, such as 'blocks.0.attn.hook_z', 'blocks.0.hook_mlp_out' or 'blocks.0.hook_resid_post'
            """
            # Get the activations that are input to the SAE
            clean_acts = cache_clean[key]
            current_grads = clean_grad[key]
            if attr_type.value == AttributionPatching.NORMAL.value:
                patched_acts = cache_patched[key]

            # Step-1: Compute the SAE latents and error terms
            sae = self.get_sae_by_hook_name(key)
            # print(f'Using {sae.cfg.hook_name} SAE for activation {key}.')

            sae_latents_act_clean = sae.encode(clean_acts)
            sae_out_clean = sae.decode(sae_latents_act_clean)
            sae_error_clean = clean_acts - sae_out_clean

            if attr_type.value == AttributionPatching.NORMAL.value:
                sae_latents_act_patched = sae.encode(patched_acts)
                sae_out_patched = sae.decode(sae_latents_act_patched)
                sae_error_patched = patched_acts - sae_out_patched

            # Step-2: Compute the gradients w.r.t. the SAE latents and error terms            
            if 'hook_z' in key: # Reshape the attn hook_z gradients to flatten the n_head dimension
                current_grads = einops.rearrange(current_grads, 
                                                'batch pos n_head d_head -> batch pos (n_head d_head)')
                sae_error_clean = einops.rearrange(sae_error_clean,
                                                'batch pos n_head d_head -> batch pos (n_head d_head)')
                if attr_type.value == AttributionPatching.NORMAL.value:
                    sae_error_patched = einops.rearrange(sae_error_patched,
                                                'batch pos n_head d_head -> batch pos (n_head d_head)')

            sae_latent_grad = einops.einsum(current_grads, sae.W_dec,
                                            'batch pos d_act, d_sae d_act -> batch pos d_sae')
            sae_error_grad = current_grads # shape [batch pos d_act]

            # Step-3 (final): Compute the score update
            if attr_type.value == AttributionPatching.NORMAL.value:
                activation_term_latents = sae_latents_act_patched - sae_latents_act_clean
                activation_term_error = sae_error_patched - sae_error_clean
            else:
                activation_term_latents = -sae_latents_act_clean
                activation_term_error = -sae_error_clean

            # SAE error term case: we want a single score per error term, so we're multiplying the d_act dimension out
            error_score_update = einops.einsum(sae_error_grad, activation_term_error,
                                               'batch pos d_act, batch pos d_act -> batch pos')
            
            # # SAE latents case: we want a score per each feature, so we're keeping the d_sae dimension
            latents_score_update = sae_latent_grad * (activation_term_latents) # shape [batch pos d_sae]

            return latents_score_update, error_score_update
        # end of compute_score_update

        for key in cache_clean.keys():
            latents_score_update, error_score_update = compute_score_update(key)
            sae_acts_post_name, sae_error_name = self.hook_name_to_sae_act_name(key)

            # print(f'Updating scores for {key} with latents_score_update of shape {latents_score_update.shape} and error_score_update of shape {error_score_update.shape}.')

            if batch_reduce == 'sum':
                latents_score_update = latents_score_update.sum(0)
                error_score_update = error_score_update.sum(0)

                node_scores[sae_acts_post_name] += latents_score_update
                node_scores[sae_error_name] += error_score_update
            elif batch_reduce == 'mean':
                latents_score_update = latents_score_update.mean(0)
                error_score_update = error_score_update.mean(0)

                node_scores[sae_acts_post_name] += latents_score_update / total_batches
                node_scores[sae_error_name] += error_score_update / total_batches

    def update_node_scores_saes_run(self, node_scores: ActivationCache, clean_grad, cache_clean, total_batches, 
                           cache_patched=None, batch_reduce='mean', attr_type=AttributionPatching.NORMAL):
        for key in node_scores.keys():
            if attr_type.value == AttributionPatching.NORMAL.value:
                activation_term = cache_patched[key] - cache_clean[key]
            elif attr_type.value == AttributionPatching.ZERO_ABLATION.value:
                # In the zero ablation variant, we set patched activations to zero
                activation_term = -cache_clean[key]

            if 'hook_sae_error' in key:
                # SAE error term case: we want a single score per error term,
                # so we're multiplying the d_act dimension out
                if 'hook_z.hook_sae_error' not in key:
                    score_update = einops.einsum(clean_grad[key], activation_term,
                                                'batch pos d_act, batch pos d_act -> batch pos')
                else:
                    score_update = einops.einsum(clean_grad[key], activation_term,
                                                'batch pos n_head d_head, batch pos n_head d_head -> batch pos')
            else:
                # SAE latents case: we want a score per each feature, so we're keeping the d_sae dimension
                score_update = clean_grad[key] * (activation_term) # shape [batch pos d_sae]

            if batch_reduce == 'sum':
                score_update = score_update.sum(0)
                node_scores[key] += score_update
            elif batch_reduce == 'mean':
                score_update = score_update.mean(0)
                node_scores[key] += score_update / total_batches

    def get_answer_logit(self, logits: Float[Tensor, "batch pos d_vocab"], clean_answers: Int[Tensor, "batch"],
                         ansnwer_pos: Int[Tensor, "batch"], return_all_logits=False) -> Float[Tensor, "batch"]:
        # clean_answers_pos_idx = clean_answers_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, logits.size(1), logits.size(2))

        answer_pos_idx = einops.repeat(ansnwer_pos, 'batch -> batch 1 d_vocab',
                                       d_vocab=logits.shape[-1])
        answer_logits = logits.gather(1, answer_pos_idx).squeeze(1) # shape [batch, d_vocab]

        correct_logits = answer_logits.gather(1, clean_answers.unsqueeze(1)).squeeze(1) # shape [batch]

        if return_all_logits:
            return answer_logits, correct_logits

        return correct_logits

    def get_logit_diff(self, logits: Float[Tensor, "batch pos d_vocab"],
                    clean_answers: Int[Tensor, "batch"], patched_answers: Int[Tensor, "batch count"],
                    answer_pos: Int[Tensor, "batch"], patch_answer_reduce='max') -> Float[Tensor, "batch"]:
        # Compute the logits for the correct answers and the tokens they have been computed at (answer_logits)
        answer_logits, correct_logits = self.get_answer_logit(logits, clean_answers, answer_pos, return_all_logits=True)

        if patched_answers.dim() == 1:  # If there's only one incorrect answer, gather the incorrect answer logits
            incorrect_logits = answer_logits.gather(1, patched_answers.unsqueeze(1)).squeeze(1)  # shape [batch]
        else:
            incorrect_logits = answer_logits.gather(1, patched_answers)  # shape [batch, answer_count]

        # If there are multiple incorrect answer options, incorrect_logits is now of shape [batch, answer_count]
        if patched_answers.dim() == 2:
            # Sum the logits for each incorrect answer option
            if patch_answer_reduce == 'sum':
                incorrect_logits = incorrect_logits.sum(dim=1)
            # Or take their maximum: this should be a better option to avoid situations where the model outputs gibberish and all the answers have similar logits
            elif patch_answer_reduce == 'max':
                incorrect_logits = incorrect_logits.max(dim=1).values

        # Both logit tensors are now of shape [batch]
        return incorrect_logits - correct_logits

    
    def _accumulate_integrated_gradients(self, cache_clean, cache_patched, grad_patched,
                                    clean_tokens, clean_answers, patched_answers,
                                    N, n_prompts, seq_len):
        raise NotImplementedError('Integrated grads are not implemented yet.')

    def get_sae_by_hook_name(self, hook_name):
        layer_num = self.hook_name_to_layer_number(hook_name)

        if 'attn.hook_z' in hook_name:
            return self.saes_dict['attn'][layer_num]
        elif 'hook_mlp_out' in hook_name:
            return self.saes_dict['mlp'][layer_num]
        elif 'hook_resid_post' in hook_name:
            return self.saes_dict['resid'][layer_num]
        else:
            raise ValueError(f'Invalid hook name: {hook_name}')
    
    def hook_name_to_sae_act_name(self, hook_name):
        # Split the input string by periods
        parts = hook_name.split('.')
        
        # Validate the input format
        if len(parts) < 3 or parts[0] != 'blocks':
            raise ValueError("Input string must start with 'blocks.<index>.'")

        # Extract the index and hook_name
        index = parts[1]
        hook_name = '.'.join(parts[2:])  # Handles cases where hook_name contains dots

        # Construct the desired SAE names
        sae_acts_post = f"blocks.{index}.{hook_name}.hook_sae_acts_post"
        sae_error = f"blocks.{index}.{hook_name}.hook_sae_error"

        return sae_acts_post, sae_error
    
    def hook_name_to_layer_number(self, hook_name):
        # Split the input string by periods
        parts = hook_name.split('.')
        
        # Validate that the string has the correct format
        if len(parts) < 3 or parts[0] != 'blocks':
            raise ValueError("Input string must start with 'blocks.<index>.'")
        
        # Extract and return the block number as an integer
        return int(parts[1])

    def key_to_d_sae(self, key):
        if 'resid' in key:
            layer_num = self.hook_name_to_layer_number(key)
            return self.resid_d_sae[layer_num]
        elif 'attn' in key:
            return self.attn_d_sae
        elif 'mlp' in key:
            return self.mlp_d_sae

    def reset_saes(self):
        self.model.reset_saes()

    def are_saes_attached(self):
        return bool(self.model.acts_to_saes)

    def add_saes(self):
        for sae in self.saes:
            self.model.add_sae(sae, use_error_term=True)

    def detach_saes_except_few(self, act_names_for_which_to_keep_saes, discard_saes=True):
        self.model.reset_saes()
        assert not self.are_saes_attached(), 'SAEs are not detached from the model.'

        original_saes_count = len(self.saes)
        # Now only load SAEs for the specified act names
        self.saes = []

        for act_name in act_names_for_which_to_keep_saes:
            try:
                sae = self.get_sae_by_hook_name(act_name)
            except ValueError as e:
                print(f'Skipping the act name {act_name} because there\'s no SAE for it. Consider reinitializing the SFC_Gemma object.')
                continue

            self.model.add_sae(sae, use_error_term=True)
            self.saes.append(sae)

        new_saes_count = len(self.saes)
        print(f'Detached {original_saes_count - new_saes_count} SAEs from the model.')

        # Discard the remaining SAEs from memory
        if discard_saes:
            try:
                del self.saes_dict
            except AttributeError:
                pass
            finally:
                clear_cache()
                print('Discarded the remaining SAEs from memory.')

        assert self.are_saes_attached(), 'SAEs should be attached to the model.'

    def print_saes(self):
        if not self.are_saes_attached():
            print('SAEs are not attached to the model.')
            return
        
        saes = self.model.acts_to_saes
        print(f'Number of SAEs: {len(saes)}')

        for name, sae in saes.items():
            print(name, sae)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _reset_sae_hooks(self):
        for sae in self.saes:
            sae.reset_hooks()

    def _load_sae(self, sae_release, sae_id):
        sae_device = self.device if self.caching_device is None else self.caching_device

        return SAE.from_pretrained(sae_release, sae_id, device=sae_device)[0].to(torch.bfloat16)
