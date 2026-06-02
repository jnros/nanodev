# enwik8 char-level DBlock for depth sweep
# defaults: L=6/B=3, 90k iters (B × 30k) — override via CLI:
#   --n_layer=8 --num_dblocks=4 --max_iters=120000 --out_dir=out-enwik8-dblock-L8-B4

out_dir = 'out-enwik8-dblock-L6-B3'
eval_interval = 1000
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'enwik8-depth-sweep'
wandb_run_name = 'dblock-L6-B3'

dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# DBlock-specific
num_dblocks = 3

learning_rate = 1e-3
max_iters = 90000
lr_decay_iters = 90000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100
