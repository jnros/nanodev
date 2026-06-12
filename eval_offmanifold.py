"""Off-manifold probe: test whether dblock denoiser is valid along ODE trajectory.

For each block, evaluates denoiser MSE at a σ grid under two input conditions:
  (i)  on-manifold:  zt = z + σ·noise   (training distribution)
  (ii) off-manifold: zt from Euler ODE trajectory integrated from σ_hi

If (i) is low everywhere but (ii) degrades as trajectory drifts from training
distribution, the blocks are only valid near their training input distribution —
not along the actual integration path.  That's a training-time finding about
the gap, not a sampler-design finding.

Output: one table per block, MSE(σ) for both conditions, σ grid on block band.

Usage:
    uv run python eval_offmanifold.py --out_dir=out-shakespeare-char-dblock-B3 \
        [--data_dir=data/shakespeare_char] [--eval_iters=50] [--batch_size=64] \
        [--n_grid=8]
"""
import argparse
import math
import os

import numpy as np
import torch

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
def probe_block(model, blk, z, pos, sigma_grid, device):
	"""
	Returns (on_mse_list, off_mse_list) each of length len(sigma_grid).

	on_mse_list[k]  = MSE of D(z + sigma_grid[k]*noise, sigma_grid[k]) to z
	off_mse_list[k] = MSE of D(zt_traj_k, sigma_grid[k]) to z
	                  where zt_traj_k is reached by Euler from sigma_grid[0]
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

	on_mse_list  = []
	off_mse_list = []

	# build trajectory: Euler from sigma_grid[0] (σ_hi) through the grid
	noise   = torch.randn_like(z)
	zt_traj = z + sigma_grid[0] * noise     # trajectory state, starts at σ_hi

	for k, sig in enumerate(sigma_grid):
		sig_t = sig.unsqueeze(0)

		# (i) on-manifold: fresh z + σ·noise
		zt_on = z + sig * noise             # same noise draw for fair comparison
		den_on = _denoise(zt_on, sig_t)
		on_mse = ((den_on - z) ** 2).mean().item()
		on_mse_list.append(on_mse)

		# (ii) off-manifold: current trajectory state
		den_off = _denoise(zt_traj, sig_t)
		off_mse = ((den_off - z) ** 2).mean().item()
		off_mse_list.append(off_mse)

		# advance trajectory by one Euler step (if not last)
		if k < len(sigma_grid) - 1:
			d1      = (zt_traj - den_off) / sig
			dt      = sigma_grid[k + 1] - sig
			zt_traj = zt_traj + d1 * dt

	return on_mse_list, off_mse_list


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

	print(f"checkpoint : {args.out_dir}  iter={ckpt['iter_num']}")
	print(f"num_dblocks: {B}   n_layer={model_args['n_layer']}")
	print(f"n_grid     : {args.n_grid} (log-uniform per block)")
	print()

	# accumulate over eval_iters batches
	on_acc  = [[0.0] * args.n_grid for _ in range(B)]
	off_acc = [[0.0] * args.n_grid for _ in range(B)]

	for _ in range(args.eval_iters):
		X, _ = get_batch(val_path, block_size, args.batch_size, args.device)
		z = model.transformer.wte(X)      # [batch, t, d]

		for blk in range(B):
			# blk 0 = high-σ (coarse); reversed
			actual = B - 1 - blk
			s_lo   = model.block_sigmas[actual]
			s_hi   = model.block_sigmas[actual + 1]
			sigma_grid = torch.linspace(math.log(s_hi), math.log(s_lo),
			                            args.n_grid,
			                            device=args.device).exp()
			on_l, off_l = probe_block(model, blk, z, pos, sigma_grid, args.device)
			for k in range(args.n_grid):
				on_acc[blk][k]  += on_l[k]
				off_acc[blk][k] += off_l[k]

	n = args.eval_iters
	for blk in range(B):
		actual = B - 1 - blk
		s_lo   = model.block_sigmas[actual]
		s_hi   = model.block_sigmas[actual + 1]
		sigma_grid = torch.linspace(math.log(s_hi), math.log(s_lo),
		                            args.n_grid,
		                            device=args.device).exp().tolist()

		print(f"block {blk} (σ {s_hi:.4f}→{s_lo:.4f})")
		print(f"  {'σ':>10}  {'on-mse':>10}  {'off-mse':>10}  {'ratio':>8}")
		for k in range(args.n_grid):
			on_m  = on_acc[blk][k]  / n
			off_m = off_acc[blk][k] / n
			ratio = off_m / on_m if on_m > 0 else float('inf')
			print(f"  {sigma_grid[k]:10.4f}  {on_m:10.5f}  {off_m:10.5f}  {ratio:8.3f}")
		print()

	print("ratio >> 1 at low-σ end → denoiser off-manifold along trajectory")
	print("ratio ≈ 1 throughout   → trajectory stays on training distribution")


if __name__ == '__main__':
	main()
