import os
import pickle
import torch
import string
import sys
from model import GPTConfig, GPT

# 1. Set your desired vocabulary size (V) via command line argument or default to 26
if len(sys.argv) > 1:
    try:
        V = int(sys.argv[1])
    except ValueError:
        print("Error: Vocabulary size V must be an integer (e.g., 6, 26, 52).")
        sys.exit(1)
else:
    V = 26  # Sensible default

# -----------------------------------------------------------------------------
checkpoint_dir = 'out_1char'      # Your 1-char checkpoint folder
data_dir = 'data/memo_1char'     # Points to your 1-character folder name!
input_path = 'input.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# -----------------------------------------------------------------------------

# 1. Define your base alphabet 
full_alphabet = (
        string.ascii_lowercase +  # a-z (26)
        string.ascii_uppercase +  # A-Z (26)
        string.digits +           # 0-9 (10)
        string.punctuation        # !, @, #, etc. (32)
    )

if V > len(full_alphabet):
    extra_needed = V - len(full_alphabet)
    extended_chars = "".join(chr(i) for i in range(161, 161 + extra_needed))
    full_alphabet += extended_chars

full_alphabet = full_alphabet.replace('=', ' ')
base_chars = full_alphabet[:V]  # Use only the first V characters based on your vocab size

# 2. Build a lookup dictionary of true pairings from input.txt
true_pairs = {}
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Could not find {input_path}. Ensure it is in this folder.")

with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        # CRITICAL FIX: Only strip the trailing newline character!
        # Do not use a generic .strip() which kills your space characters.
        line = line.rstrip('\n')
        if '=' in line:
            prompt_side, target_side = line.split('=')
            # Retain potential spaces inside the actual token representations
            true_pairs[prompt_side] = target_side

# 3. Generate all 2-character prompt combinations directly
test_prompts = [c1 + c2 for c1 in base_chars for c2 in base_chars]

# 4. Load your 1-Character Tokenizer mappings (meta.pkl)
meta_path = os.path.join(data_dir, 'meta.pkl')
if not os.path.exists(meta_path):
    raise FileNotFoundError(f"Could not find meta.pkl at {meta_path}. Check data_dir path!")

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']

# Helper lambda to turn tokens back into readable text strings
decode = lambda l: ''.join([itos[i] for i in l])

# 5. Load the trained 1-character model from your checkpoint folder
ckpt_path = os.path.join(checkpoint_dir, 'ckpt_30000.pt')
print(f"Loading 1-Character checkpoint from {ckpt_path}...")
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']

# Clean up any potential DDP compiler prefixes
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.eval()
model.to(device)

# 6. Evaluate all pairings
correct_count = 0
total_count = len(test_prompts)
failed_samples = []  # Quietly tracks failures instead of flooding the screen

print(f"\nRunning 1-Character Accuracy Evaluation (Model Vocab Size: {gptconf.vocab_size})...")
print("-" * 60)

for idx, prompt in enumerate(test_prompts):
    # Print a clean, in-place progress update every 100 prompts
    if idx % 100 == 0 or idx == total_count - 1:
        sys.stdout.write(f"\rProcessing: {idx + 1}/{total_count} prompts...")
        sys.stdout.flush()

    eval_prompt = f"{prompt}="
    
    try:
        start_ids = [stoi[c] for c in eval_prompt]
    except KeyError as e:
        print(f"\nTokenizer Error: Could not find character key {e} in 1-char mappings.")
        sys.exit(1)

    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
    max_new_tokens = 2 
    
    with torch.no_grad():
        y = model.generate(x, max_new_tokens, temperature=0.0001, top_k=5)
    
    full_output = decode(y[0].tolist())
    
    if '=' in full_output:
        predicted_target = full_output.split('=')[1]
    else:
        predicted_target = full_output.replace(prompt, "").replace("=", "")
        
    predicted_target = predicted_target.replace('\n', '')
    ground_truth = true_pairs[prompt]
    
    if predicted_target == ground_truth:
        correct_count += 1
        # Show the first 5 successes as structural confirmation
        if correct_count <= 5:
            print(f"\n[SAMPLE SUCCESS] {prompt}={predicted_target} (Expected: {ground_truth})")
    else:
        # Save failures to print in a controlled way at the end
        failed_samples.append((prompt, predicted_target, ground_truth))

# 7. Print Final Summary Metrics and Controlled Failure Log
print("\n" + "-" * 60)
accuracy_pct = (correct_count / total_count) * 100
print(f"1-CHAR MODEL TOTAL ACCURACY: {accuracy_pct:.2f}% ({correct_count}/{total_count} Correct)")
print("-" * 60)

# Display a tidy log of up to 10 failures so you can inspect errors without buffer cutoff
if failed_samples:
    print(f"Tracking {len(failed_samples)} total failures. Displaying a subset of errors:")
    for f_prompt, f_pred, f_truth in failed_samples[:10]:
        print(f"[FAIL] {f_prompt}={f_pred} (Expected: {f_truth})")
    if len(failed_samples) > 10:
        print(f"   ... and {len(failed_samples) - 10} more failures hidden to preserve terminal buffer.")
else:
    print("Zero failures logged.")
print("-" * 60)