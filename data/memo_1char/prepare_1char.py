import os
import pickle
import sys
import string
import numpy as np

def build_vocabulary(vocab_length: int):
    """
    Recreates the exact V-length alphabet pool logic and integrates
    necessary control characters dynamically.
    """
    # 1. Build core pool matching character generation
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

    # Ensure equal sign doesn't create duplicate space bugs
    character_pool = character_pool.replace('=', ' ')
    alphabet = list(character_pool[:vocab_length])

    # 2. Add structural control tokens explicitly and sort
    all_tokens = sorted(list(set(alphabet + ['=', '\n', ' '])))
    
    stoi = {ch: i for i, ch in enumerate(all_tokens)}
    itos = {i: ch for i, ch in enumerate(all_tokens)}
    
    return stoi, itos, len(all_tokens)

def main():
    # --- 1. Hyperparameters & CLI Parsing ---
    vocab_length = 26  # Default fallback
    if len(sys.argv) > 1:
        try:
            vocab_length = int(sys.argv[1])
        except ValueError:
            print("Error: Vocab length argument must be an integer (e.g., 26, 52, 128).")
            sys.exit(1)

    current_dir = os.path.dirname(__file__)
    input_file_path = "../../input.txt"  # Relative path to the input dataset

    if not os.path.exists(input_file_path):
        print(f"Error: Could not find training data file at {input_file_path}")
        sys.exit(1)

    # --- 2. Build Mappings ---
    stoi, itos, vocab_size = build_vocabulary(vocab_length)
    print(f"Total vocabulary size (Alphabet + Control Chars): {vocab_size}")

    # --- 3. Stream & Process Dataset ---
    print("Reading dataset into memory...")
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    total_chars = len(data)
    print(f"Length of dataset in characters: {total_chars:,}")

    # --- 4. High-Efficiency Vectorized Encoding ---
    print("Encoding text stream to tokens...")
    # Find the maximum unicode code point value among all keys
    max_unicode_val = max(ord(char) for char in stoi.keys())
    lookup_arr = np.zeros(max_unicode_val + 1, dtype=np.uint16)
    for char, idx in stoi.items():
        lookup_arr[ord(char)] = idx

    # Convert the entire raw string data into an array of unicode code points, then map
    encoded_ids = lookup_arr[np.frombuffer(data.encode('utf-8'), dtype=np.uint8)]

    # --- 5. Split Dataset ---
    split_idx = int(total_chars * 0.9)
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

    print("Successfully saved train.bin, val.bin, and meta.pkl!")

if __name__ == '__main__':
    main()