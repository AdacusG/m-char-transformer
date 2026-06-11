# Transformer Tokenization vs. Compute Scaling Study

An empirical machine learning scaling study investigating the performance and hardware resource trade-offs between data engineering (tokenization strategies) and algorithmic brute-forcing (parameter scaling) using a small-scale Generative Pre-trained Transformer (GPT) architecture.

## Core Experimental Results

| Tokenizer Type | Base Vocab ($V$) | Unique Pairs | Configuration | Parameters (M) | Training Time (s) | Final Loss | Accuracy (%) | VRAM Usage (GB) | 
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1-Char** | 6 | 36 | 4, 4, 128 | 0.79M | 44.58 |0.8452 | **100.00%** | 0.085 |
| **2-Char** | 6 | 36 | 4, 4, 128 | 0.79M | 44.13 |0.8073 | **100.00%** | 0.085 |
| **1-Char** | 26 | 676 | 4, 4, 128 | 0.79M | 46.35 |2.2708 | **0.00%** | 0.085 |
| **2-Char** | 26 | 676 | 4, 4, 128 | 0.87M | 44.57 |1.8018 | **99.85%** | 0.087 |
| **1-Char** | 52 | 2,704 | 4, 4, 128 | 0.79M | 45.59 |2.7478 | **0.00%** | 0.085 |
| **2-Char** | 52 | 2,704 | 4, 4, 128 | 1.13M | 45.68 |2.2940 | **99.82%** | 0.102 |
| **1-Char (Upgraded)** | 52 | 2,704 | **6, 8, 256** | **4.7M** | ~180.00 |**1.8400** | **98.37%** | 0.162 |

## Key Insights

1. **The Character Bottleneck:** At larger vocabulary scales ($V \ge 26$), standard character-level tokenization introduces steep multi-token sequence dependencies that cause small transformer architectures to experience complete capacity collapse (0.00% accuracy).
2. **The Data Engineering Advantage:** Shifting to a 2-Character bigram tokenizer compresses sequence lengths and offloads the data relationships directly into the vocabulary embedding tables. Keeping the base `4, 4, 128` model layout, it achieves **99.82% accuracy** effortlessly.
3. **The Compute Scaling Premium:** Resolving the sequence bottleneck while forcing a 1-Char approach requires scaling the network to a heavy `6, 8, 256` setup (**4.7M parameters**). While this rescues accuracy (**98.37%**), it introduces a massive hardware premium: **over 4x the training duration and 59% more peak GPU memory utilization** compared to the tokenization approach.

## Requirements
To run the code, you need the following requesites, based upon nanoGPT:
```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```
**IF YOU WANT TO USE A CPU:** You must change `device=cuda` in `train.py`, `evaluate_1char.py`, and `evaluate_2char.py`.

**IF YOU PLAN ON USING A CUDA GPU:** First, you must run the following command so that pytorch is on the correct version:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
**Note:** The end of the URL might be different based upon what Nvidia GPU you own, to make sure this version is not to new, run the following command to get the CUDA driver version:
```
nvidia-smi
```

## Getting Started & Replication

### 1. Generate Raw Data
Generate the uniform sequence mapping equations in the root folder, and then copy the `input.txt` file that is generated into the `data/memo_1char` and `data/memo_2char` folders:
```
python gen_memo_data.py [vocab_size]
```
### 2. Prepare Data for Training
In both `memo_#char` folders, run the following commands to prepare the data for training:
```
python prepare_1char.py
```
```
python prepare_2char.py
```
### 3. Train the Models
Within the root directory, run the following code to train the models:
```
python train.py config/train_1char.py
```
```
python train.py config/train_2char.py
```
After training these models, there will be an output of the final loss, training time, time-per-=setp, and VRAM usage.
### 4. Test Accuracy of Models
To test the accuracy of each model, run the following code from the root directory:
```
python evaluate_1char.py [vocab_size]
```
```
python evaluate_2char.py [vocab_size]
```
**NOTE:** The vocab_size given here must match the vocab_size used in the data generation.
