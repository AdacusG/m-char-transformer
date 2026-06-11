import itertools
import string
import random
import sys
def gen_memo_data(vocabLength, repeatTimes, fileName):
    character_pool = (
        string.ascii_lowercase +  # a-z (26)
        string.ascii_uppercase +  # A-Z (26)
        string.digits +           # 0-9 (10)
        string.punctuation        # !, @, #, etc. (32)
    )
    character_pool = character_pool.replace('=', ' ') # Space is used to replace '=' in the mapping as it is a special character.
    # 2. Check if we need to dynamically inject extra characters
    if vocabLength > len(character_pool):
        extra_needed = vocabLength - len(character_pool)
        # Pull safe, unique extended characters starting from index 161
        extended_chars = "".join(chr(i) for i in range(161, 161 + extra_needed))
        character_pool += extended_chars

    character_pool = character_pool[:vocabLength]
    alphabet = list(character_pool)
    # Inputs and Outputs are direct copys, where they are all 3 letter combinations of the given alphabet.
    inputs = ["".join(p) for p in itertools.product(alphabet, repeat=3)]
    outputs = inputs.copy()
    # Randomized outputs, but with a fixed seed for reproducibility.
    random.seed(42)
    random.shuffle(outputs)
    # Create the unique pairs of inputs and outputs, and then repeat them accordingly, and later shuffle all data afterwards.
    unique_pairs = []
    for input, output in zip(inputs, outputs):
        unique_pairs.append(f"{input}={output}\n")
    dataset = unique_pairs * repeatTimes
    random.shuffle(dataset)
    # Write to file.
    with open(fileName, "w", encoding="utf-8") as f:
        f.writelines(dataset)
    print(f"Generated {len(unique_pairs)} unique mappings.")
if __name__ == "__main__":
    gen_memo_data(int(sys.argv[1]), 100, "input.txt")