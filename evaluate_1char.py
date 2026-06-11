import os
import pickle
import torch
import string
import sys
from model import GPTConfig, GPT

# --- 1. Arguments & Configuration ---
if len(sys.argv) > 1:
    try:
        V = int(sys.argv[1])
        checkpoint = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
    except ValueError:
        print("Error: Vocabulary size V must be an integer (e.g., 6, 26, 52).")
        sys.exit(1)
else:
    V = 26

checkpoint_dir = 'out_1char'
data_dir = 'data/memo_1char'
input_path = 'input.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Adjust BATCH_SIZE based on your string length (n). 
# For n=2 or n=3, a batch size of 1024 or 2048 will slide right into your 16GB VRAM.
BATCH_SIZE = 1024  

# --- 2. Build Base Alphabet & Clean Delimiters ---
full_alphabet = (
    string.ascii_lowercase +
    string.ascii_uppercase +
    string.digits +
    string.punctuation
)

if V > len(full_alphabet):
    extra_needed = V - len(full_alphabet)
    extended_chars = "".join(chr(i) for i in range(161, 161 + extra_needed))
    full_alphabet += extended_chars

full_alphabet = full_alphabet.replace('=', ' ')
base_chars = full_alphabet[:V]

# --- 3. Build Truth Map & Generate Target Prompts ---
true_pairs = {}
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Could not find {input_path}.")

with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.rstrip('\n')
        if '=' in line:
            prompt_side, target_side = line.split('=')
            true_pairs[prompt_side] = target_side

# Generate evaluation configurations
test_prompts = [c1 + c2 for c1 in base_chars for c2 in base_chars]
total_count = len(test_prompts)

# --- 4. Load Mappings & Model Checkpoint ---
meta_path = os.path.join(data_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']

ckpt_path = os.path.join(checkpoint_dir, f'ckpt_{checkpoint}.pt')
print(f"Loading 1-Character checkpoint from {ckpt_path}...")
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']

unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.eval()
model.to(device)

# Using compile here cuts execution graph overhead down dramatically!
model = torch.compile(model)

print(f"\nRunning Parallel Evaluation (Batch Size: {BATCH_SIZE})...")
print("-" * 60)

correct_count = 0
failed_samples = []
sample_successes = []

# --- 5. Batched Evaluation Loop ---
for i in range(0, total_count, BATCH_SIZE):
    batch_prompts = test_prompts[i:i + BATCH_SIZE]
    
    # Vectorized tokenization construction across the batch
    tokenized_batch = []
    for prompt in batch_prompts:
        eval_prompt = f"{prompt}="
        tokenized_batch.append([stoi[c] for c in eval_prompt])
        
    # Shape: (batch_size, prompt_length)
    x = torch.tensor(tokenized_batch, dtype=torch.long, device=device)
    max_new_tokens = 2 # Change this dynamically to matches your target 'n' length later!
    
    with torch.no_grad():
        # Batched forward generation pass
        y = model.generate(x, max_new_tokens, temperature=0.0001, top_k=5)
    
    # Parse results out of the generated batch matrices
    generated_sequences = y.tolist()
    for idx, seq in enumerate(generated_sequences):
        full_output = ''.join([itos[token] for token in seq])
        prompt = batch_prompts[idx]
        
        predicted_target = full_output.split('=')[1].replace('\n', '')
        ground_truth = true_pairs[prompt]
        
        if predicted_target == ground_truth:
            correct_count += 1
            if len(sample_successes) < 5:
                sample_successes.append((prompt, predicted_target, ground_truth))
        else:
            failed_samples.append((prompt, predicted_target, ground_truth))

    sys.stdout.write(f"\rProcessed: {min(i + BATCH_SIZE, total_count)}/{total_count} prompts..." + "\n")
    sys.stdout.flush()

# --- 6. Results & Visual Diagnostic Log ---
accuracy_pct = (correct_count / total_count) * 100
print(f"1-CHAR MODEL TOTAL ACCURACY: {accuracy_pct:.2f}% ({correct_count}/{total_count} Correct)")
if failed_samples:
    print(f"Tracking {len(failed_samples)} total failures. Displaying a subset of errors:")
    for f_prompt, f_pred, f_truth in failed_samples[:10]:
        print(f"[FAIL] {f_prompt}={f_pred} (Expected: {f_truth})")
    if len(failed_samples) > 10:
        print(f"   ... and {len(failed_samples) - 10} more failures hidden.")