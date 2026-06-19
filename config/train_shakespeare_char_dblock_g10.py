# DiffusionBlocks AR on shakespeare-char, Sakana overlap gamma=0.1 (text default)
# Identical to train_shakespeare_char_dblock.py except gamma + out_dir.

out_dir = 'out-shakespeare-char-dblock-B3-g10'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'dblock-B3-g10'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# DBlock-specific
num_dblocks = 3
gamma = 0.1  # Sakana overlap factor (text default); smooths block boundaries

learning_rate = 1e-3
max_iters = 30000
lr_decay_iters = 30000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100
