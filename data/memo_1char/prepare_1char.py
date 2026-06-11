import os
import pickle
import sys
import string  # <-- Make sure string is imported!
import numpy as np

# 1. Read your generated text file
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
print(f"Length of dataset in characters: {len(data):,}")

# 2. Re-create your exact V=128 alphabet pool logic
if len(sys.argv) > 1:
    try:
        vocabLength = int(sys.argv[1])
    except ValueError:
        print("Error: Vocab length argument must be an integer (ie 26, 52, 128).")
        sys.exit(1)
else:
    vocabLength = 26  # Fallback default

character_pool = (
    string.ascii_lowercase +
    string.ascii_uppercase +
    string.digits +
    string.punctuation
)

if vocabLength > len(character_pool):
    extra_needed = vocabLength - len(character_pool)
    extended_chars = "".join(chr(i) for i in range(161, 161 + extra_needed))
    character_pool += extended_chars

# CRITICAL FIX: Replace '=' with ' ' inside the pool before slicing
# This mirrors your data generation logic exactly
character_pool = character_pool.replace('=', ' ')

alphabet = list(character_pool[:vocabLength])

# 3. CRITICAL: Add your control characters to the alphabet
# The dataset contains strings like "ab=yx\n" (where '=' is now ' '), so we track them explicitly!
chars = sorted(list(set(alphabet + ['=', '\n', ' '])))
vocab_size = len(chars)

print(f"Total vocabulary size (Alphabet + Control Chars): {vocab_size}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s]
def decode(l):
    return ''.join([itos[i] for i in l])

# --- Rest of the script remains exactly the same ---
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

train_ids = encode(train_data)
val_ids = encode(val_data)

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Successfully saved train.bin, val.bin, and meta.pkl!")