"""ODE discretization diagnostic.

Loads a dblock checkpoint, sweeps n_steps (NFE per block) over the val set,
prints per-block MSE + final-block CE vs n_steps.

Diagnostic readout:
  - CE/MSE decreasing within the sweep → ODE discretization is a real gap source
  - Gains concentrated in low-σ (last) blocks → fine-detail blocks are the bottleneck
  - Flat curve → gap is bias (target mismatch / block composition), not discretization

NOTE: the baseline row uses the chained random-σ forward (different eval protocol).
      It is NOT comparable to the sweep rows; do not compute deltas against it.
      The meaningful comparison is internal to the sweep: n_steps=1 vs n_steps=32.

Usage:
    uv run python eval_multistep.py --out_dir=out-shakespeare-char-dblock-B3 \
        [--data_dir=data/shakespeare_char] [--eval_iters=200] [--batch_size=64] \
        [--steps 1 2 4 8 16 32]
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
	p.add_argument('--data_dir',   default=None,
	               help='defaults to data/<dataset> from checkpoint config')
	p.add_argument('--eval_iters', type=int, default=200)
	p.add_argument('--batch_size', type=int, default=64)
	p.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
	p.add_argument('--steps', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
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
def eval_standard(model, data_path, block_size, batch_size, device, eval_iters):
	"""Chained random-σ forward — different protocol from sweep rows."""
	model.eval()
	ces = []
	for _ in range(eval_iters):
		X, Y = get_batch(data_path, block_size, batch_size, device)
		_, _, ce = model(X, Y)
		ces.append(ce.item())
	return sum(ces) / len(ces)


@torch.no_grad()
def eval_sweep(model, data_path, block_size, batch_size, device, eval_iters, n_steps):
	"""Per-block MSE + final-block CE using ODE Euler trajectory."""
	model.eval()
	B = model.num_dblocks
	ce_acc   = 0.0
	mse_acc  = [0.0] * B
	for _ in range(eval_iters):
		X, Y = get_batch(data_path, block_size, batch_size, device)
		ce, mse_list = model.forward_multistep_eval(X, Y, n_steps=n_steps)
		ce_acc  += ce.item()
		for b, m in enumerate(mse_list):
			mse_acc[b] += m.item()
	n = eval_iters
	return ce_acc / n, [m / n for m in mse_acc]


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

	print(f"checkpoint : {args.out_dir}  iter={ckpt['iter_num']}")
	print(f"dataset    : {dataset}  val={val_path}")
	print(f"num_dblocks: {B}   n_layer={model_args['n_layer']}")
	print(f"σ bands    : {['%.4f–%.4f' % (model.block_sigmas[B-1-b], model.block_sigmas[B-b]) for b in range(B)]}")
	print()

	# baseline (different protocol — do NOT compare deltas against sweep rows)
	base_ce = eval_standard(model, val_path, block_size, args.batch_size,
	                        args.device, args.eval_iters)

	# header
	mse_hdrs = '  '.join(f'blk{b}_mse' for b in range(B))
	print(f"{'n_steps':>8}  {'final_ce':>9}  {mse_hdrs}")
	print(f"{'baseline*':>8}  {base_ce:9.4f}  {'(chained random-σ, different protocol)':}")
	print()

	for n in args.steps:
		ce, mses = eval_sweep(model, val_path, block_size, args.batch_size,
		                      args.device, args.eval_iters, n)
		mse_cols = '  '.join(f'{m:9.5f}' for m in mses)
		print(f"{n:>8}  {ce:9.4f}  {mse_cols}")

	print()
	print("* baseline: chained random-σ forward; not comparable to sweep rows.")
	print("  Diagnostic: is the sweep CE/MSE monotonically decreasing with n_steps?")


if __name__ == '__main__':
	main()
