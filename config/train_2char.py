# Train a tiny "pico" GPT on our memorization task
# Run it from the root nanoGPT directory with: python train.py config/train_2char.py

out_dir = 'out_2char'
eval_interval = 1000 # Evaluate the model every 500 steps
eval_iters = 20
log_interval = 10 # Log training loss every 10 steps

# Always save checkpoints if the validation loss improves
always_save_checkpoint = True

dataset = 'memo_2char'
gradient_accumulation_steps = 1
batch_size = 256 
block_size = 8 # Our sequences are short (e.g., "ab=db\n" is ~6 chars), 8 is plenty

# Roughly ~0.79M parameters, not including the embedding layer.
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

learning_rate = 1e-3 # Fast learning rate for simple tasks
max_iters = 30000     # How many total training steps to run
lr_decay_iters = 30000 # When to start decaying the learning rate (i.e., slow down training)
min_lr = 1e-4        # lr_decay_iters == max_iters means learning rate decays to min at the end
warmup_iters = 100
backend = 'gloo' # change to 'gloo' if on Windows, but often not needed for single CPU/GPU
compile = False
