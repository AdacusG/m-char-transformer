import os
import pickle
import sys
import string
import numpy as np

def build_vocabulary(vocab_length: int):
    character_pool = (
        string.ascii_lowercase +
        string.ascii_uppercase +
        string.digits +
        string.punctuation
    )

    if vocab_length > len(character_pool):
        extra_needed = vocab_length - len(character_pool)
        extended_chars = "".join(chr(i) for i in range(161, 161 + extra_needed))
        character_pool += extended_chars

    character_pool = character_pool.replace('=', ' ')
    alphabet = list(character_pool[:vocab_length])
    
    # 1. Create pure bigrams separate from your base characters to prevent list loops
    bigrams = [c1 + c2 for c1 in alphabet for c2 in alphabet]
    
    # 2. Prevent sorting from scrambling your structural token logic
    all_tokens = alphabet + bigrams + ['=', '\n']
    all_tokens = sorted(list(set(all_tokens)))
    
    stoi = {token: i for i, token in enumerate(all_tokens)}
    itos = {i: token for i, token in enumerate(all_tokens)}
    
    return stoi, itos, len(all_tokens)

def main():
    vocab_length = 26
    if len(sys.argv) > 1:
        try:
            vocab_length = int(sys.argv[1])
        except ValueError:
            print("Error: Vocab length argument must be an integer.")
            sys.exit(1)

    current_dir = os.path.dirname(__file__)
    input_file_path = "../../input.txt" # Adjusted to match your text directory location

    if not os.path.exists(input_file_path):
        print(f"Error: Could not find training data file at {input_file_path}")
        sys.exit(1)

    # --- 2. Build Mappings ---
    stoi, itos, vocab_size = build_vocabulary(vocab_length)
    print(f"Total vocabulary size (M-Char Matrix Space): {vocab_size}")

    # --- 3. Stream & Process Dataset ---
    print("Reading dataset into memory...")
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    total_chars = len(data)
    print(f"Length of dataset in characters: {total_chars:,}")

    # --- 4. GREEDY M-CHAR ENCODER LOOP ---
    print("Encoding text stream using Greedy Longest-Match First...")
    encoded_ids = []
    i = 0
    
    while i < total_chars:
        # Step A: Look ahead to see if a two-character bigram is possible
        if i + 1 < total_chars:
            potential_bigram = data[i:i+2]
            if potential_bigram in stoi:
                encoded_ids.append(stoi[potential_bigram])
                i += 2  # Skip forward two full positions
                continue
                
        # Step B: Fallback to single character tokenization if no bigram matches
        current_char = data[i]
        if current_char in stoi:
            encoded_ids.append(stoi[current_char])
        else:
            # Safety backup: handle unexpected characters cleanly
            print(f"Warning: Character {repr(current_char)} at index {i} missing from vocabulary!")
        i += 1

    # Convert the python tracking list directly to an optimized NumPy binary array
    encoded_ids = np.array(encoded_ids, dtype=np.uint16)
    print(f"Tokenization complete. Generated {len(encoded_ids):,} total tokens.")

    # --- 5. Split Dataset ---
    total_tokens = len(encoded_ids)
    split_idx = int(total_tokens * 0.9)
    train_ids = encoded_ids[:split_idx]
    val_ids = encoded_ids[split_idx:]

    # --- 6. Save Artifacts to Disk ---
    print("Writing binaries to disk...")
    train_ids.tofile(os.path.join(current_dir, 'train.bin'))
    val_ids.tofile(os.path.join(current_dir, 'val.bin'))

    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(current_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print("Successfully saved mixed-token train.bin, val.bin, and meta.pkl!")

if __name__ == '__main__':
    main()