import os
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import matplotlib.pyplot as plt
from model import GPTConfig, GPT

# --- 1. Default Configuration Space ---
out_dir = 'out'
eval_interval = 2000
log_interval = 100  # Shifted from 1 to 100 to eliminate terminal print bottlenecks!
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# Dataset & Architecture Settings
dataset = 'openwebtext'
gradient_accumulation_steps = 16  # Set to simulate larger effective batches safely
batch_size = 32                  # Micro-batch size optimized for your GPU VRAM
block_size = 1024

# Hyperparameters (Optimized for small-to-mid transformers)
n_layer, n_head, n_embd = 12, 12, 768
dropout = 0.0
bias = False
learning_rate = 6e-4
max_iters = 30000
weight_decay = 1e-1
beta1, beta2 = 0.9, 0.95
grad_clip = 1.0

# Learning Rate Decay Settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 30000
min_lr = 6e-5

# System Setup
backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# --- 2. Configurator Overrides ---
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # Dynamic command line flag integration
config = {k: globals()[k] for k in config_keys}

# --- 3. Distributed & Hardware Initialization ---
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    device = f'cuda:{int(os.environ["LOCAL_RANK"])}'
    torch.cuda.set_device(device)
    master_process = int(os.environ['RANK']) == 0
    seed_offset = int(os.environ['RANK'])
    assert gradient_accumulation_steps % int(os.environ['WORLD_SIZE']) == 0
    gradient_accumulation_steps //= int(os.environ['WORLD_SIZE'])
else:
    master_process = True
    seed_offset = 0

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# --- 4. High-Efficiency Data Streaming Stream ---
data_dir = os.path.join('data', dataset)
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)

# --- 5. Model Assembly ---
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"Found dataset vocab_size = {meta_vocab_size} inside {meta_path}")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=meta_vocab_size if meta_vocab_size is not None else 50304, dropout=dropout)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if compile:
    print("Compiling model graph via PyTorch 2.0...")
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])])

# --- 6. Helper Verification Pipelines ---
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def print_diagnostic_report(milestone_step, elapsed_time, current_loss):
    """Generates an explicit training diagnostic report at milestone windows."""
    steps_tracked = 9900 if milestone_step == 10000 else 10000
    seconds_per_step = elapsed_time / steps_tracked
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    
    print("\n" + "="*50)
    print(f"📊 TRAINING BENCHMARK REPORT @ STEP {milestone_step}")
    print("="*50)
    print(f"Target Vocabulary Size ($V$) : {meta_vocab_size}")
    print(f"Milestone Step Cross Loss    : {current_loss:.4f}")
    print(f"Window Evaluation Time       : {elapsed_time:.2f} seconds")
    print(f"Average Speed                : {seconds_per_step:.4f} seconds/step")
    print(f"Peak GPU Memory Used         : {peak_mem_gb:.3f} GB")
    print("="*50 + "\n")

# --- 7. Core Training Loop Execution ---
iter_num = 0
best_val_loss = 1e9
iterations_history, loss_history = [], []
bench_start_time = None

X, Y = get_batch('train')
t0 = time.time()
raw_model = model.module if ddp else model

while iter_num <= max_iters:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Loss Evaluations & Dynamic Checking
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        iterations_history.append(iter_num)
        loss_history.append(losses['train'])
        
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                if iter_num in [10000, 20000, 30000]:
                    milestone_filename = f'ckpt_{iter_num}.pt'
                    torch.save(checkpoint, os.path.join(out_dir, milestone_filename))
                    print(f"Saved custom milestone checkpoint: {milestone_filename}")
                else:
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    # Forward / Backward Update Step Block
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

    # --- 8. Diagnostic Tracking Windows ---
    if master_process:
        # Step 100: Set Baseline
        if iter_num == 100:
            torch.cuda.synchronize()
            bench_start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            print("Warmup complete. Benchmarking tracking active...")

        # Step 10,000 Milestone
        elif iter_num == 10000:
            torch.cuda.synchronize()
            lossf = loss.item() * gradient_accumulation_steps
            print_diagnostic_report(10000, time.time() - bench_start_time, lossf)
            bench_start_time = time.time()  # Reset clock anchor for next run

        # Step 20,000 Milestone
        elif iter_num == 20000:
            torch.cuda.synchronize()
            lossf = loss.item() * gradient_accumulation_steps
            print_diagnostic_report(20000, time.time() - bench_start_time, lossf)
            bench_start_time = time.time()  # Reset clock anchor for final run

        # Step 30,000 Milestone
        elif iter_num == 30000:
            torch.cuda.synchronize()
            lossf = loss.item() * gradient_accumulation_steps
            print_diagnostic_report(30000, time.time() - bench_start_time, lossf)

    iter_num += 1

# --- 9. Post-Training Metrics Generation ---
if master_process:
    print("Training complete! Generating loss graph...")
    plt.figure(figsize=(10, 6))
    plt.plot(iterations_history, loss_history, color='blue', linewidth=1.5, label='Train Loss')
    plt.title(f'Training Loss over Iterations ({out_dir})', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Iterations', fontsize=12, labelpad=10)
    plt.ylabel('Loss', fontsize=12, labelpad=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    
    plot_path = f"{out_dir}/loss_curve.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Graph successfully saved to {plot_path}")

if ddp:
    destroy_process_group()