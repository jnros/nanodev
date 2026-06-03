# enwik8 interconnect experiment: L=6, B=2 (one block per GPU, W=2)
# 60k iters = B * 30k (same per-block compute as enwik8 depth sweep)
#
# override sync_interval on CLI:
#   --sync_interval=100 --out_dir=out-interconnect-sync100

out_dir = 'out-interconnect-sync1'
eval_interval = 2000
eval_iters = 200
log_interval = 50

always_save_checkpoint = False

wandb_log = False
wandb_project = 'interconnect'
wandb_run_name = 'sync1'

dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

num_dblocks = 2

learning_rate = 1e-3
max_iters = 60000
lr_decay_iters = 60000
min_lr = 1e-4
warmup_iters = 2000

sync_interval = 1
