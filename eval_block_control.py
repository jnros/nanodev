"""Control: confine forward_chained_eval to ONE block's band (isolated self-drift).

Generalizes eval_block2_control.py to any layer-block. If the chaining machinery,
restricted so only block TGT fires, lands near that block's single-block
teacher-forced CE, then forward_chained_eval is faithful and the composition cost
is real, not a harness artifact.

layer-block TGT (high σ → block 0) owns band index bi = num_dblocks-1-TGT into the
ascending block_sigmas: band = [block_sigmas[bi], block_sigmas[bi+1]].

Usage:
  uv run python eval_block_control.py <ckpt> [target_block]
"""
import os, sys, numpy as np, torch
from model import GPTConfig
from model_dblock import GPTDBlock

device      = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir    = 'data/shakespeare_char'
block_size  = 256
batch_size  = 64
eval_iters  = 50
seed        = 1337
num_dblocks = 3

DBLOCK_CKPT = (sys.argv[1] if len(sys.argv) > 1
               else os.environ.get('DBLOCK_CKPT',
                                   'out-shakespeare-char-dblock-B3/ckpt.pt'))
TGT = int(sys.argv[2]) if len(sys.argv) > 2 else num_dblocks - 1

val = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def make_batches():
	g = torch.Generator().manual_seed(seed)
	out = []
	for _ in range(eval_iters):
		ix = torch.randint(len(val) - block_size, (batch_size,), generator=g)
		x = torch.stack([torch.from_numpy(val[i:i+block_size].astype(np.int64)) for i in ix])
		y = torch.stack([torch.from_numpy(val[i+1:i+1+block_size].astype(np.int64)) for i in ix])
		out.append((x.to(device), y.to(device)))
	return out

print(f"loading checkpoint: {DBLOCK_CKPT}  target_block={TGT}")
ck = torch.load(DBLOCK_CKPT, map_location=device, weights_only=False)
args = dict(ck['model_args']); args['dropout'] = 0.0
db = GPTDBlock(GPTConfig(**args), num_dblocks=num_dblocks)
sd = ck['model']
for k in list(sd.keys()):
	if k.startswith('_orig_mod.'):
		sd[k[len('_orig_mod.'):]] = sd.pop(k)
db.load_state_dict(sd)
db = db.to(device).eval()

batches = make_batches()
bi   = num_dblocks - 1 - TGT
s_lo = db.block_sigmas[bi]
s_hi = db.block_sigmas[bi + 1] * 0.999      # stay inside the band → only TGT fires
print(f"block {TGT} band: sigma_min={s_lo:.4f}  sigma_max={s_hi:.4f}\n")

for n in (2, 4, 8, 16):
	torch.manual_seed(seed)
	ces, blocks_seen = [], None
	for X, Y in batches:
		out = db.forward_chained_eval(X, Y, n_steps=n, solver='euler',
		                              grid='global', from_noise=False,
		                              sigma_min=s_lo, sigma_max=s_hi)
		ces.append(out['ce']); blocks_seen = out['blocks']
	uniq = sorted(set(blocks_seen))
	print(f"block{TGT}-only n={n:<3d}: CE={np.mean(ces):.4f}  blocks_fired={uniq}")
