"""Off-manifold probe: test whether dblock denoiser is valid along ODE trajectory.

For each block, evaluates denoiser MSE at a σ grid under two input conditions:
  (i)  on-manifold:  zt = z + σ·noise   (training distribution)
  (ii) off-manifold: zt from Euler ODE trajectory integrated from σ_hi

If (i) is low everywhere but (ii) degrades as trajectory drifts from training
distribution, the blocks are only valid near their training input distribution —
not along the actual integration path.  That's a training-time finding about
the gap, not a sampler-design finding.

--chain threads the off-manifold trajectory ACROSS block boundaries: block k+1
starts integrating from block k's exit state, not from a fresh z + σ·noise.
This measures cross-block error inheritance — the actual chained-inference path.
Default (off) self-seeds each block from clean data (within-block self-drift).

Output: one table per block, MSE(σ) for both conditions, σ grid on block band.
Written to eval_offmanifold_{chain,nochain}_{out_dir}.txt and to stdout.

Usage:
    uv run python eval_offmanifold.py --out_dir=out-shakespeare-char-dblock-B3 \
        [--data_dir=data/shakespeare_char] [--eval_iters=50] [--batch_size=64] \
        [--n_grid=8] [--chain]
"""
import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from model import GPTConfig
from model_dblock import GPTDBlock


def parse_args():
	p = argparse.ArgumentParser()
	p.add_argument('--out_dir',    required=True)
	p.add_argument('--data_dir',   default=None)
	p.add_argument('--eval_iters', type=int, default=50)
	p.add_argument('--batch_size', type=int, default=64)
	p.add_argument('--n_grid',     type=int, default=8,
	               help='σ grid points per block (log-uniform)')
	p.add_argument('--chain',      action='store_true',
	               help='thread off-manifold trajectory across blocks '
	                    '(cross-block inheritance); default self-seeds per block')
	p.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
	return p.parse_args()


def get_batch(data_path, block_size, batch_size, device):
	data = np.memmap(data_path, dtype=np.uint16, mode='r')
	ix   = torch.randint(len(data) - block_size, (batch_size,))
	x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64))
	                 for i in ix])
	y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64))
	                 for i in ix])
	return x.to(device), y.to(device)


@torch.no_grad()
def probe_block(model, blk, z, pos, sigma_grid, device, targets, zt_in=None):
	"""
	Returns (on_mse_list, off_mse_list, on_ce_list, off_ce_list, zt_out).

	on_mse_list[k]  = MSE of D(z + sigma_grid[k]*noise, sigma_grid[k]) to z
	off_mse_list[k] = MSE of D(zt_traj_k, sigma_grid[k]) to z
	                  where zt_traj_k is reached by Euler integration.
	on_ce_list[k]   = CE of lm_head(D(on))  vs targets  (training CE path,
	off_ce_list[k]  = CE of lm_head(D(off)) vs targets    model_dblock.py:200-208)
	zt_out          = trajectory state at the block's low-σ exit (= next
	                  block's σ_hi); feed back as zt_in to chain blocks.

	zt_in is None  -> self-seed trajectory from z + sigma_grid[0]*noise
	zt_in is state -> inherit upstream block's exit state (chained inference)

	NOTE: only the LAST block's low-σ row is comparable to baseline/chained CE;
	intermediate high-σ rows decode coarse estimates (model_dblock.py:227).
	"""
	B = model.num_dblocks
	b, t, d = z.shape

	def _denoise(zt, sigma_scalar):
		sv = sigma_scalar.expand(b)
		s2  = sv ** 2
		sd2 = model.sigma_data ** 2
		c_s = sd2 / (s2 + sd2)
		c_o = sv * model.sigma_data / (s2 + sd2).sqrt()
		c_i = 1.0 / (s2 + sd2).sqrt()
		c_n = 0.25 * sv.log()
		x = zt * c_i[:, None, None] + model.transformer.wpe(pos)
		for i in model.layer_assignment[blk]:
			x = model.transformer.h[i](x, c_n)
		x = model.transformer.ln_f(x)
		return x * c_o[:, None, None] + zt * c_s[:, None, None]

	def _ce(den):
		logits = model.lm_head(den)
		return F.cross_entropy(logits.view(-1, logits.size(-1)),
		                       targets.view(-1)).item()

	on_mse_list  = []
	off_mse_list = []
	on_ce_list   = []
	off_ce_list  = []

	# on-manifold comparison always uses fresh noise on true z
	noise = torch.randn_like(z)
	# off-manifold trajectory: inherit upstream exit state if chaining,
	# else self-seed from z + σ_hi·noise (born on training distribution)
	zt_traj = z + sigma_grid[0] * noise if zt_in is None else zt_in

	for k, sig in enumerate(sigma_grid):
		sig_t = sig.unsqueeze(0)

		# (i) on-manifold: fresh z + σ·noise
		zt_on = z + sig * noise             # same noise draw for fair comparison
		den_on = _denoise(zt_on, sig_t)
		on_mse_list.append(((den_on - z) ** 2).mean().item())
		on_ce_list.append(_ce(den_on))

		# (ii) off-manifold: current trajectory state
		den_off = _denoise(zt_traj, sig_t)
		off_mse_list.append(((den_off - z) ** 2).mean().item())
		off_ce_list.append(_ce(den_off))

		# advance trajectory by one Euler step (if not last)
		if k < len(sigma_grid) - 1:
			d1      = (zt_traj - den_off) / sig
			dt      = sigma_grid[k + 1] - sig
			zt_traj = zt_traj + d1 * dt

	return on_mse_list, off_mse_list, on_ce_list, off_ce_list, zt_traj


@torch.no_grad()
def main():
	args = parse_args()

	ckpt = torch.load(os.path.join(args.out_dir, 'ckpt.pt'),
	                  map_location=args.device)
	model_args = ckpt['model_args']
	cfg = GPTConfig(**model_args)
	num_dblocks = ckpt['config'].get('num_dblocks', 3)
	model = GPTDBlock(cfg, num_dblocks=num_dblocks)
	state = ckpt['model']
	prefix = '_orig_mod.'
	for k in list(state.keys()):
		if k.startswith(prefix):
			state[k[len(prefix):]] = state.pop(k)
	model.load_state_dict(state)
	model.to(args.device)
	model.eval()

	cfg_dict   = ckpt['config']
	dataset    = cfg_dict.get('dataset', 'shakespeare_char')
	data_dir   = args.data_dir or os.path.join('data', dataset)
	val_path   = os.path.join(data_dir, 'val.bin')
	block_size = model_args['block_size']
	B          = model.num_dblocks
	pos        = torch.arange(block_size, device=args.device)

	# collect output for both stdout and file
	lines = []
	def emit(s=''):
		print(s)
		lines.append(s)

	mode = 'chained' if args.chain else 'unchained (self-seed per block)'
	emit(f"checkpoint : {args.out_dir}  iter={ckpt['iter_num']}")
	emit(f"num_dblocks: {B}   n_layer={model_args['n_layer']}")
	emit(f"n_grid     : {args.n_grid} (log-uniform per block)")
	emit(f"mode       : {mode}")
	emit()

	# accumulate over eval_iters batches
	on_acc      = [[0.0] * args.n_grid for _ in range(B)]
	off_acc     = [[0.0] * args.n_grid for _ in range(B)]
	on_ce_acc   = [[0.0] * args.n_grid for _ in range(B)]
	off_ce_acc  = [[0.0] * args.n_grid for _ in range(B)]

	for _ in range(args.eval_iters):
		X, Y = get_batch(val_path, block_size, args.batch_size, args.device)
		z = model.transformer.wte(X)      # [batch, t, d]

		zt_in = None                       # block 0 always self-seeds
		for blk in range(B):
			# blk 0 = high-σ (coarse); reversed
			actual = B - 1 - blk
			s_lo   = model.block_sigmas[actual]
			s_hi   = model.block_sigmas[actual + 1]
			sigma_grid = torch.linspace(math.log(s_hi), math.log(s_lo),
			                            args.n_grid,
			                            device=args.device).exp()
			on_l, off_l, on_ce_l, off_ce_l, zt_out = probe_block(
			    model, blk, z, pos, sigma_grid, args.device, Y, zt_in)
			for k in range(args.n_grid):
				on_acc[blk][k]     += on_l[k]
				off_acc[blk][k]    += off_l[k]
				on_ce_acc[blk][k]  += on_ce_l[k]
				off_ce_acc[blk][k] += off_ce_l[k]
			# chained: next block inherits this block's exit state;
			# unchained: next block self-seeds from clean data
			zt_in = zt_out if args.chain else None

	n = args.eval_iters
	for blk in range(B):
		actual = B - 1 - blk
		s_lo   = model.block_sigmas[actual]
		s_hi   = model.block_sigmas[actual + 1]
		sigma_grid = torch.linspace(math.log(s_hi), math.log(s_lo),
		                            args.n_grid,
		                            device=args.device).exp().tolist()

		emit(f"block {blk} (σ {s_hi:.4f}→{s_lo:.4f})")
		emit(f"  {'σ':>10}  {'on-mse':>10}  {'off-mse':>10}  {'ratio':>8}"
		     f"  {'on-ce':>9}  {'off-ce':>9}")
		for k in range(args.n_grid):
			on_m  = on_acc[blk][k]     / n
			off_m = off_acc[blk][k]    / n
			on_c  = on_ce_acc[blk][k]  / n
			off_c = off_ce_acc[blk][k] / n
			ratio = off_m / on_m if on_m > 0 else float('inf')
			emit(f"  {sigma_grid[k]:10.4f}  {on_m:10.5f}  {off_m:10.5f}  {ratio:8.3f}"
			     f"  {on_c:9.4f}  {off_c:9.4f}")
		emit()

	emit("ratio >> 1 at low-σ end → denoiser off-manifold along trajectory")
	emit("ratio ≈ 1 throughout   → trajectory stays on training distribution")
	emit("CE: only LAST block low-σ row comparable to baseline/chained-eval CE")
	emit("    (intermediate high-σ rows decode coarse estimates; nats, vocab-dependent)")

	tag      = 'chain' if args.chain else 'nochain'
	base     = os.path.basename(args.out_dir.rstrip('/'))
	out_path = f"eval_offmanifold_{tag}_{base}.txt"
	with open(out_path, 'w') as f:
		f.write('\n'.join(lines) + '\n')
	print(f"\nwrote {out_path}")


if __name__ == '__main__':
	main()
