"""Assemble a composed dblock ckpt from per-block 2b fine-tunes.

Each layer-block's transformer.h.* layers come from ITS OWN ckpt; everything else
(blocks not overridden + shared wte/wpe/ln_f/lm_head) comes from --base. No
cross-block weight mixing within a block — independence preserved. Writes
<out_dir>/ckpt.pt for the composed eval (eval_offmanifold.py --chain).

layer-block b owns layers [b*split, (b+1)*split), split = n_layer // num_dblocks.

Usage:
  uv run python assemble_2b.py --base out-shakespeare-char-dblock-B3 \
     --out_dir out-shakespeare-char-dblock-B3-2b-all --num_dblocks 3 \
     --block 1=out-shakespeare-char-dblock-B3-2b-blk1 \
     --block 2=out-shakespeare-char-dblock-B3-2b
"""
import os, argparse, torch

def load_sd(path):
	if os.path.isdir(path):
		path = os.path.join(path, 'ckpt.pt')
	ck = torch.load(path, map_location='cpu', weights_only=False)
	sd = ck['model']
	for k in list(sd.keys()):
		if k.startswith('_orig_mod.'):
			sd[k[len('_orig_mod.'):]] = sd.pop(k)
	return sd, ck

p = argparse.ArgumentParser()
p.add_argument('--base', required=True, help='dir/ckpt for block 0 + shared params')
p.add_argument('--out_dir', required=True)
p.add_argument('--num_dblocks', type=int, default=3)
p.add_argument('--block', action='append', default=[],
               help='IDX=path  (repeatable): override layer-block IDX from path')
args = p.parse_args()

base_sd, base_ck = load_sd(args.base)
margs = dict(base_ck['model_args'])
n_layer = margs['n_layer']
split = n_layer // args.num_dblocks
assert split * args.num_dblocks == n_layer, "n_layer not divisible by num_dblocks"

out = dict(base_sd)            # start from base (block 0 + shared)
provenance = {b: 'base' for b in range(args.num_dblocks)}
for spec in args.block:
	idx_s, path = spec.split('=', 1)
	b = int(idx_s)
	src_sd, _ = load_sd(path)
	layers = range(b * split, (b + 1) * split)
	n_copied = 0
	for i in layers:
		pref = f"transformer.h.{i}."
		for k in list(out.keys()):
			if k.startswith(pref):
				assert k in src_sd, f"missing {k} in {path}"
				out[k] = src_sd[k]
				n_copied += 1
	provenance[b] = path
	print(f"block {b}  layers={list(layers)}  copied {n_copied} tensors from {path}")

# sanity: shapes match
for k in out:
	assert out[k].shape == base_sd[k].shape, f"shape drift at {k}"

os.makedirs(args.out_dir, exist_ok=True)
torch.save({'model': out, 'model_args': margs, 'iter_num': -1,
            'config': {'assembled_from': provenance, 'split': split}},
           os.path.join(args.out_dir, 'ckpt.pt'))
print(f"\nassembled → {args.out_dir}/ckpt.pt")
for b in range(args.num_dblocks):
	print(f"  block {b}: {provenance[b]}")
