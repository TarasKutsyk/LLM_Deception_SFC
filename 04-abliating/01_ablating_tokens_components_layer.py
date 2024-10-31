# %%
from transformer_lens import HookedTransformer, utils

# Load the model
model = HookedTransformer.from_pretrained("gpt2-small")

# Define the ablation hook with closure
def create_ablation_hook(positions_to_ablate):
    def ablation_hook(activation, hook):
        # Set the activation at specified positions to 0
        activation[:, positions_to_ablate, :] = 0
        return activation
    return ablation_hook

# Specify what to ablate
layers_to_ablate = [0, 1, 2]  
positions_to_ablate = [3, 4]  # Token positions to ablate
components_to_ablate = ["attn_out", "mlp_out"]  # Ablate both attention and MLP outputs

# Create hooks list
hooks = []
for layer in layers_to_ablate:
    for component in components_to_ablate:
        hook_name = utils.get_act_name(component, layer)
        # Create a specific hook function for these positions
        hook_fn = create_ablation_hook(positions_to_ablate)
        hooks.append((hook_name, hook_fn))

# Run model without ablation first
input_text = "The quick brown fox jumps over the lazy dog"
tokens = model.to_tokens(input_text)
logits_original = model(tokens)

# Get the probability for "dog" token in original output
# Change this to identify the tokens to ablate by token name and hte whole vocab
dog_token_id = tokens[0, -1].item()  # Get the token logits for "dog"
original_probs = logits_original[0, -2].softmax(dim=-1)  # Probs for token before "dog". Be aware I already write the last token. Therefore I look in the token before the last.
original_dog_prob = original_probs[dog_token_id].item()

# Run with hooks to ablate components
logits_ablated = model.run_with_hooks(
    tokens,
    fwd_hooks=hooks
)

# Get probability for "dog" token in ablated output
ablated_probs = logits_ablated[0, -2].softmax(dim=-1) # Probs for token before "dog". Be aware I already write the last token. Therefore I look in the token before the last.
ablated_dog_prob = ablated_probs[dog_token_id].item()

# Print comparison results
print("Original text:", input_text)
print(f"Probability of 'dog' token:")
print(f"  - Before ablation: {original_dog_prob:.4f}")
print(f"  - After ablation:  {ablated_dog_prob:.4f}")
print(f"  - Relative change: {((ablated_dog_prob - original_dog_prob) / original_dog_prob) * 100:.1f}%")

# Optional: Print top predictions before and after
top_k = 5
print(f"\nTop {top_k} predictions before ablation:")
top_tokens_original = original_probs.topk(top_k)
for prob, token_id in zip(top_tokens_original.values, top_tokens_original.indices):
    print(f"  {model.to_string(token_id):15} : {prob:.4f}")

print(f"\nTop {top_k} predictions after ablation:")
top_tokens_ablated = ablated_probs.topk(top_k)
for prob, token_id in zip(top_tokens_ablated.values, top_tokens_ablated.indices):
    print(f"  {model.to_string(token_id):15} : {prob:.4f}")
# %%
