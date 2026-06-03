"""Interconnect experiment: DiffusionBlocks with one block per device.

Block-specific params (transformer.h layers) never synced across devices.
Shared params (wte, wpe, ln_f, lm_head) synced every sync_interval steps.
Measures how often shared embeddings need to communicate for blocks to
remain coherent — the decoupling tolerance of block-wise diffusion training.

Usage (single node, 2 GPUs, PCIe bandwidth only):
    NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 \\
        train_dblock_interconnect.py config/train_interconnect.py \\
        --sync_interval=100 --out_dir=out-interconnect-sync100

Two-node:
    torchrun --nproc_per_node=1 --nnodes=2 \\
        --rdzv_backend=c10d --rdzv_endpoint=<host>:29500 ...

Rank r owns layer_assignment[r]. sigma sampling is rank-local (samples
only from rank's σ band). Shared param sync is the only cross-device
communication during training.
"""
import math
import os
import pickle
import json
import time
import types
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist

from model import GPTConfig
from model_dblock import GPTDBlock, _norm_cdf

# -----------------------------------------------------------------------------
out_dir = 'out'
eval_interval = 2000
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = False
dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False
learning_rate = 1e-3
max_iters = 60000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 60000
min_lr = 1e-4
backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False  # compile + method monkeypatch don't mix
num_dblocks = 2  # must equal world_size (one block per GPU)
sync_interval = 1  # shared param sync every N optimizer steps; large N = less comm
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items()
               if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

assert int(os.environ.get('RANK', -1)) != -1, \
    "must launch via torchrun (RANK env var not set)"

dist.init_process_group(backend=backend)
rank       = dist.get_rank()
local_rank = int(os.environ['LOCAL_RANK'])
world_size = dist.get_world_size()
device     = f'cuda:{local_rank}'
torch.cuda.set_device(device)
master = rank == 0

assert num_dblocks == world_size, \
    f"num_dblocks ({num_dblocks}) must equal world_size ({world_size}); " \
    f"one block per device"
my_block = rank  # rank r owns layer_assignment[r]

# same seed everywhere → same data on all ranks (controlled comparison)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
           'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)

if master:
    os.makedirs(out_dir, exist_ok=True)

data_dir = os.path.join('data', dataset)

def get_batch(split):
	data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
	ix = torch.randint(len(data) - block_size, (batch_size,))
	x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
	y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
	return (x.pin_memory().to(device, non_blocking=True),
	        y.pin_memory().to(device, non_blocking=True))

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
	with open(meta_path, 'rb') as f:
		meta = pickle.load(f)
	meta_vocab_size = meta['vocab_size']
	if master:
		print(f"vocab_size = {meta_vocab_size}")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                  block_size=block_size, bias=bias,
                  vocab_size=meta_vocab_size or 50304, dropout=dropout)
gptconf = GPTConfig(**model_args)
model = GPTDBlock(gptconf, num_dblocks=num_dblocks).to(device)


# ---------------------------------------------------------------------------
# per-rank σ sampler
#
# layer_assignment[r] handles HIGH σ for r=0 (first layers) and LOW σ for
# r=B-1 (last layers).  _target_block reverses the σ ordering, so rank r
# must sample from block_sigmas index (num_dblocks-1-r).
# ---------------------------------------------------------------------------

def _make_block_sampler(m, blk_idx):
	sigma_range_idx = m.num_dblocks - 1 - blk_idx
	s_lo = m.block_sigmas[sigma_range_idx]
	s_hi = m.block_sigmas[sigma_range_idx + 1]
	pm, ps = -1.2, 1.2
	c_lo = _norm_cdf((math.log(s_lo) - pm) / ps)
	c_hi = _norm_cdf((math.log(s_hi) - pm) / ps)
	# rank-local generator: σ samples independent across ranks.
	# data loading keeps the shared global RNG for controlled batch comparison.
	_gen = torch.Generator()
	_gen.manual_seed(1337 + blk_idx * 10007)  # wide spacing; hash() avoided (PYTHONHASHSEED)

	def _sample(self, n, dev):
		u = torch.rand(n, generator=_gen)
		p = c_lo + (c_hi - c_lo) * u
		ppf = math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)
		sigma = torch.exp(torch.tensor(pm, dtype=torch.float32)
		                  + torch.tensor(ps, dtype=torch.float32) * ppf)
		return sigma.to(dev)

	return types.MethodType(_sample, m)


_original_sample_sigmas = model._sample_sigmas.__func__  # unbound for restore
model._sample_sigmas = _make_block_sampler(model, my_block)

# ---------------------------------------------------------------------------
# shared vs block param split
# ---------------------------------------------------------------------------

_SHARED = ('transformer.wte.', 'transformer.wpe.',
           'transformer.ln_f.', 'lm_head.')

def _is_shared(name):
	return any(name.startswith(p) for p in _SHARED)

def sync_shared_params():
	"""Average shared params across all ranks."""
	for name, param in model.named_parameters():
		if _is_shared(name):
			dist.all_reduce(param.data, op=dist.ReduceOp.AVG)

def assemble_all_blocks():
	"""Each rank r broadcasts its block layers to all ranks for eval."""
	for blk_r in range(world_size):
		for layer_i in model.layer_assignment[blk_r]:
			for param in model.transformer.h[layer_i].parameters():
				dist.broadcast(param.data, src=blk_r)

# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss():
	# sync shared, gather all block params → rank 0 has full model
	sync_shared_params()
	assemble_all_blocks()
	out = {}
	if master:
		# restore random-block sampler for full-model eval
		model._sample_sigmas = types.MethodType(_original_sample_sigmas, model)
		model.eval()
		for split in ['train', 'val']:
			edm_losses = torch.zeros(eval_iters)
			ce_losses  = torch.zeros(eval_iters)
			for k in range(eval_iters):
				X, Y = get_batch(split)
				with ctx:
					_, edm_loss, ce_loss = model(X, Y)
				edm_losses[k] = edm_loss.item()
				ce_losses[k]  = ce_loss.item()
			out[split]         = edm_losses.mean()
			out[f'{split}_ce'] = ce_losses.mean()
		model.train()
		# restore rank-specific sampler
		model._sample_sigmas = _make_block_sampler(model, my_block)
	return out

# ---------------------------------------------------------------------------
# optimizer
# ---------------------------------------------------------------------------

scaler    = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate,
                                       (beta1, beta2), 'cuda')

def get_lr(it):
	if it < warmup_iters:
		return learning_rate * (it + 1) / (warmup_iters + 1)
	if it > lr_decay_iters:
		return min_lr
	ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
	coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
	return min_lr + coeff * (learning_rate - min_lr)

# ---------------------------------------------------------------------------
# σ diagnostic: verify disjoint ranges and independent sampling across ranks
# ---------------------------------------------------------------------------

def _sigma_diagnostic(n=100):
	"""Sample n σ values, gather to rank 0, print ranges and inter-rank correlation."""
	sigma = model._sample_sigmas(n, device)
	gathered = [torch.zeros(n, device=device) for _ in range(world_size)]
	dist.all_gather(gathered, sigma)
	gathered = [s.cpu() for s in gathered]
	if not master:
		return
	bs = model.block_sigmas
	lines = ["--- σ diagnostic ---"]
	for r, s in enumerate(gathered):
		ri = model.num_dblocks - 1 - r  # sigma_range_idx for rank r
		s_lo, s_hi = bs[ri], bs[ri + 1]
		in_range = ((s >= s_lo) & (s <= s_hi)).float().mean().item()
		lines.append(f"  rank {r}: range=[{s_lo:.4f},{s_hi:.4f}]  "
		             f"min={s.min():.4f} max={s.max():.4f} "
		             f"in_range={in_range:.2f}  "
		             f"first3={[round(x,4) for x in s[:3].tolist()]}")
	normed = []
	for r, s in enumerate(gathered):
		ri = model.num_dblocks - 1 - r
		log_lo = math.log(bs[ri])
		log_hi = math.log(bs[ri + 1])
		normed.append((torch.log(s) - log_lo) / (log_hi - log_lo))
	r_pearson = torch.corrcoef(torch.stack(normed))[0, 1].item()
	lines.append(f"  Pearson r (log-normalized, ≈ underlying u): {r_pearson:.4f}  "
	             f"({'OK' if abs(r_pearson) < 0.1 else 'CORRELATED — check RNG'})")
	lines.append("--- end σ diagnostic ---")
	diag_path = os.path.join(out_dir, 'sigma_diagnostic.txt')
	with open(diag_path, 'w') as f:
		f.write('\n'.join(lines) + '\n')
	print('\n'.join(lines))  # also print for live runs

_sigma_diagnostic()

# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------

if master:
	print(f"rank {rank} owns block {my_block}, "
	      f"layers {model.layer_assignment[my_block]}, "
	      f"sync_interval={sync_interval}")

torch.cuda.reset_peak_memory_stats()

X, Y           = get_batch('train')
t0             = time.time()
iter_num       = 0
best_val_loss  = 1e9
best_val_ce    = 1e9
loss_log       = []

while True:
	lr = get_lr(iter_num) if decay_lr else learning_rate
	for pg in optimizer.param_groups:
		pg['lr'] = lr

	if iter_num % eval_interval == 0:
		losses = estimate_loss()
		if master:
			print(f"step {iter_num}: "
			      f"train {losses['train']:.4f} (ce {losses['train_ce']:.4f}), "
			      f"val {losses['val']:.4f} (ce {losses['val_ce']:.4f})")
			peak_mb = torch.cuda.max_memory_allocated() / 1024**2
			entry = {
				'iter':         iter_num,
				'train':        losses['train'].item(),
				'val':          losses['val'].item(),
				'train_ce':     losses['train_ce'].item(),
				'val_ce':       losses['val_ce'].item(),
				'peak_vram_mb': peak_mb,
			}
			loss_log.append(entry)
			if losses['val'] < best_val_loss:
				best_val_loss = losses['val'].item()
				best_val_ce   = losses['val_ce'].item()

	if iter_num == 0 and eval_only:
		break

	for micro_step in range(gradient_accumulation_steps):
		with ctx:
			_, loss, _ = model(X, Y)
			loss = loss / gradient_accumulation_steps
		X, Y = get_batch('train')
		scaler.scale(loss).backward()

	if grad_clip != 0.0:
		scaler.unscale_(optimizer)
		torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
	scaler.step(optimizer)
	scaler.update()
	optimizer.zero_grad(set_to_none=True)

	if (iter_num + 1) % sync_interval == 0:
		sync_shared_params()

	t1 = time.time()
	dt = t1 - t0
	t0 = t1
	if iter_num % log_interval == 0 and master:
		print(f"iter {iter_num}: loss {loss.item() * gradient_accumulation_steps:.4f}, "
		      f"{dt*1000:.1f}ms")
	iter_num += 1

	if iter_num > max_iters:
		break

dist.destroy_process_group()

if master:
	peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
	summary = {
		'scenario':        'interconnect',
		'dataset':         dataset,
		'n_layer':         n_layer,
		'n_head':          n_head,
		'n_embd':          n_embd,
		'block_size':      block_size,
		'batch_size':      batch_size,
		'num_dblocks':     num_dblocks,
		'sync_interval':   sync_interval,
		'max_iters':       max_iters,
		'best_val_loss':   best_val_loss,
		'best_val_ce':     best_val_ce,
		'peak_vram_mb':    peak_vram_mb,
	}
	with open(os.path.join(out_dir, 'dblock_summary.json'), 'w') as f:
		json.dump(summary, f, indent=2)
	with open(os.path.join(out_dir, 'loss_curves.json'), 'w') as f:
		json.dump(loss_log, f, indent=2)
	print(f"peak VRAM: {peak_vram_mb:.1f} MB")
	print(f"saved to {out_dir}")
