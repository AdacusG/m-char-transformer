import os
import pickle
import sys
import torch
from model import GPTConfig, GPT

# --- 1. Configuration & CLI Parsing ---
V = int(sys.argv[1]) if len(sys.argv) > 1 else 26
checkpoint_num = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
torch.set_float32_matmul_precision('high')

checkpoint_dir = 'out_mchar'
data_dir = 'data/memo_mchar'
input_path = 'input.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1024  

# --- 2. Load Metadata and Model Checkpoint ---
meta_path = os.path.join(data_dir, 'meta.pkl')
if not os.path.exists(meta_path):
    raise FileNotFoundError(f"Could not find metadata at {meta_path}. Please run prepare_mchar.py first.")

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']

ckpt_path = os.path.join(checkpoint_dir, f'ckpt_{checkpoint_num}.pt')
print(f"Loading M-Char checkpoint from {ckpt_path}...")
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']

# Clean up potential DDP compilation prefixes
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.eval()
model.to(device)
#model = torch.compile(model)

# --- 3. Build Ground Truth Validation Map ---
true_pairs = {}
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Could not find validation text dataset at {input_path}.")

with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.rstrip('\n')
        if '=' in line:
            prompt_side, target_side = line.split('=')
            true_pairs[prompt_side] = target_side

# Extract all evaluation prompt baselines (e.g., 'abc')
test_prompts = list(true_pairs.keys())
total_count = len(test_prompts)

# --- 4. Define Specialized Forced Pathway Tokenizers (No Padding) ---
# These functions now return ONLY the true tokens for that pathway.
def tokenize_1_1_1(prompt, stoi):
    # Forces separate characters: ['c', 'a', 'i', '='] (Length 4)
    return [stoi[prompt[0]], stoi[prompt[1]], stoi[prompt[2]], stoi['=']]

def tokenize_2_1(prompt, stoi):
    # Forces a bigram then a character: ['ca', 'i', '='] (Length 3)
    return [stoi[prompt[0:2]], stoi[prompt[2]], stoi['=']]

def tokenize_1_2(prompt, stoi):
    # Forces a character then a bigram: ['c', 'ai', '='] (Length 3)
    return [stoi[prompt[0]], stoi[prompt[1:3]], stoi['=']]

def tokenize_natural(prompt, stoi):
    """
    Greedy Longest-Match First tokenizer that mimics how 
    the training text was naturally built.
    """
    tokens = []
    # Combine the prompt and the delimiter to match the full input prefix
    full_prefix = prompt + '='
    total_chars = len(full_prefix)
    i = 0
    
    while i < total_chars:
        # Check for bigram match first
        if i + 1 < total_chars:
            potential_bigram = full_prefix[i:i+2]
            if potential_bigram in stoi:
                tokens.append(stoi[potential_bigram])
                i += 2
                continue
        
        # Fallback to single character/structural token
        current_char = full_prefix[i]
        if current_char in stoi:
            tokens.append(stoi[current_char])
        i += 1
        
    return tokens

pathways = {
    "1-1-1 Pathway (Forced Atomic)": tokenize_1_1_1,
    "2-1   Pathway (Greedy Natural)": tokenize_2_1,
    "1-2   Pathway (Alternative)   ": tokenize_1_2,
    "Natural Unforced Pathway      ": tokenize_natural
}

print(f"\nEvaluating {total_count:,} prompts across 3 distinct tokenization pathways...")
print("=" * 65)

# --- 5. Multi-Pathway Execution Loop ---
for pathway_name, tokenize_fn in pathways.items():
    correct_count = 0
    
    for i in range(0, total_count, BATCH_SIZE):
        batch_prompts = test_prompts[i:i + BATCH_SIZE]
        
        tokenized_batch = []
        for prompt in batch_prompts:
            tokenized_batch.append(tokenize_fn(prompt, stoi))
            
        x = torch.tensor(tokenized_batch, dtype=torch.long, device=device)
        
        # --- Character-Aware Generation Loop ---
        # Start with the prompt context
        generated = x # Shape: (B, T)
        
        # Max tokens to generate as a fallback safety
        for _ in range(4): 
            with torch.no_grad():
                logits, _ = model(generated)
                # Pluck the logits for the final position, apply temperature
                logits = logits[:, -1, :] / 0.01
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
        
        # Decode and strictly evaluate characters
        generated_sequences = generated.tolist()
        for idx, seq in enumerate(generated_sequences):
            prompt = batch_prompts[idx]
            
            # Find where the prompt ends and generation begins
            input_len = len(tokenized_batch[idx])
            gen_tokens = seq[input_len:]
            
            # Decode the newly generated tokens into a string step-by-step
            predicted_target = ""
            for token in gen_tokens:
                char_val = itos[token]
                if char_val == '\n':
                    break
                predicted_target += char_val
                # STRICT CAP: Stop as soon as we have our 3 target characters!
                if len(predicted_target) >= 3:
                    predicted_target = predicted_target[:3]
                    break
            
            ground_truth = true_pairs[prompt].strip()
            
            if predicted_target == ground_truth:
                correct_count += 1
                
    accuracy_pct = (correct_count / total_count) * 100
    print(f"{pathway_name} Accuracy: {accuracy_pct:6.2f}%  ({correct_count}/{total_count})")

print("=" * 65)