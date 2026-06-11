# Train a tiny "pico" GPT on our memorization task (Scenario A - Character-level)
# Run it from the root nanoGPT directory with: python train.py config/train_mchar.py
init_from = "resume" # 'scratch' or 'resume'
out_dir = 'out_mchar'
eval_interval = 1000   # Evaluate the model every 1000 steps
eval_iters = 20
log_interval = 100     # Log training loss every 100 steps

# Always save checkpoints if the validation loss improves
always_save_checkpoint = True

dataset = 'memo_mchar'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 8        # Our sequences are short (e.g., "ab=db\n" is ~6 chars), 8 is plenty

# Roughly ~0.79M parameters.
n_layer = 4
n_head = 4 
n_embd = 128
dropout = 0.0

learning_rate = 1e-3  # Fast learning rate for simple tasks
max_iters = 30000 # How many total training steps to run
lr_decay_iters = 30000 # When to start decaying the learning rate 
min_lr = 1e-4 # Minimum learning rate after decay
warmup_iters = 100
backend = 'gloo'      # Safe backend for Windows environments
compile = False