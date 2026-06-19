"""Off-shell conditioning: fine-tune ONE block in isolation to extend its valid
region. --mode rollout = 2b (self-rollout drift states, THE FIX); --mode sigma =
2a (σ-mismatch, dead negative result). 2b is the one we use.

Fine-tune ONE block (the lowest-σ band, block num_dblocks-1) in isolation. Every
step targets that block. 2b (rollout): input = the block's OWN realized ODE
states. 2a (sigma): with prob `off_frac` the input is pushed OFF the
{x0 + σ·eps} shell by labelling a more-noised embedding with a smaller σ
(σ_noise = σ_label · m, m≥1). Target is always the clean tokens. The other
blocks and all shared params (wte/wpe/ln_f/lm_head) are FROZEN — block trained
alone, no cross-block coupling.

Success metric is external: run eval_block2_control.py against the saved ckpt
and check the self-drift CE-vs-n climb flattens (esp. mid-σ). Pre-registered:
expect mid-σ help, possible low-σ stall (EDM skip → c_out→0 gives the net no
leverage to correct off-shell error there).

Usage (2b self-rollout, the one we use):
  uv run python finetune_offshell.py --mode rollout \
      --init_ckpt out-shakespeare-char-dblock-B3-g10/ckpt.pt \
      --out_dir   out-shakespeare-char-dblock-B3-g10-2b-blk2 \
      --gamma 0.1 --target_block 2 --max_iters 500
"""
import os, math, json, argparse, numpy as np, torch
from model import GPTConfig
from model_dblock import GPTDBlock

p = argparse.ArgumentParser()
p.add_argument('--init_ckpt', default='out-shakespeare-char-dblock-B3/ckpt.pt')
p.add_argument('--out_dir',   default='out-shakespeare-char-dblock-B3-2a')
p.add_argument('--data_dir',  default='data/shakespeare_char')
p.add_argument('--num_dblocks', type=int, default=3)
p.add_argument('--gamma', type=float, default=0.0)
p.add_argument('--target_block', type=int, default=None,
               help='layer-block to fine-tune (high σ→0). default = lowest-σ block')
p.add_argument('--max_iters',   type=int, default=5000)
p.add_argument('--ema_decay',   type=float, default=0.9,
               help='EMA over eval points for the drift-CE flatten/early-stop signal')
p.add_argument('--patience',    type=int, default=8,
               help='early-stop: eval points w/o EMA-drift improvement (0=disable)')
p.add_argument('--min_delta',   type=float, default=1e-3,
               help='min EMA-drift improvement to reset patience')
p.add_argument('--lr',          type=float, default=1e-4)
p.add_argument('--mode', choices=['sigma', 'rollout'], default='sigma',
               help='sigma=2a (σ-mismatch); rollout=2b (self-rollout drift states)')
p.add_argument('--off_frac',    type=float, default=0.5,
               help='2a: fraction of each batch made off-shell (σ-mismatch)')
p.add_argument('--mismatch_max', type=float, default=3.0,
               help='2a: σ_noise = σ_label·m, m~logU[1,mismatch_max] for off-shell rows')
p.add_argument('--rollout_steps', type=int, default=8,
               help='2b: ODE steps in the self-rollout (states trained on)')
p.add_argument('--batch_size',  type=int, default=64)
p.add_argument('--block_size',  type=int, default=256)
p.add_argument('--eval_interval', type=int, default=250)
p.add_argument('--log_interval', type=int, default=20,
               help='heartbeat: print train-loss EMA every N iters')
p.add_argument('--eval_iters',  type=int, default=50)
p.add_argument('--seed',        type=int, default=1337)
p.add_argument('--device',      default='cuda' if torch.cuda.is_available() else 'cpu')
args = p.parse_args()

torch.manual_seed(args.seed)
os.makedirs(args.out_dir, exist_ok=True)
device = args.device

# --- data ---
train = np.memmap(os.path.join(args.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val   = np.memmap(os.path.join(args.data_dir, 'val.bin'),   dtype=np.uint16, mode='r')

def get_batch(split):
	d  = train if split == 'train' else val
	ix = torch.randint(len(d) - args.block_size, (args.batch_size,))
	x  = torch.stack([torch.from_numpy(d[i:i+args.block_size].astype(np.int64)) for i in ix])
	y  = torch.stack([torch.from_numpy(d[i+1:i+1+args.block_size].astype(np.int64)) for i in ix])
	return x.to(device), y.to(device)

# --- model ---
ck = torch.load(args.init_ckpt, map_location=device, weights_only=False)
margs = dict(ck['model_args']); margs['dropout'] = 0.0
model = GPTDBlock(GPTConfig(**margs), num_dblocks=args.num_dblocks, gamma=args.gamma)
sd = ck['model']
for k in list(sd.keys()):
	if k.startswith('_orig_mod.'):
		sd[k[len('_orig_mod.'):]] = sd.pop(k)
model.load_state_dict(sd)
model = model.to(device)

# layer-block TGT (high σ → block 0) owns band index bi = num_dblocks-1-TGT into
# the ascending block_sigmas: band = [block_sigmas[bi], block_sigmas[bi+1]].
TGT        = args.target_block if args.target_block is not None else args.num_dblocks - 1
bi         = args.num_dblocks - 1 - TGT
tgt_layers = model.layer_assignment[TGT]
s_lo, s_hi = model.block_sigmas[bi], model.block_sigmas[bi + 1]
s_cap      = model.block_sigmas[-1]
print(f"fine-tuning block {TGT}  layers={tgt_layers}  band=[{s_lo:.4f},{s_hi:.4f}]")

# freeze everything except the target block's transformer layers
trainable = set()
for i in tgt_layers:
	for n, _ in model.transformer.h[i].named_parameters():
		trainable.add(f"transformer.h.{i}.{n}")
for n, pm in model.named_parameters():
	pm.requires_grad = n in trainable
n_train = sum(pm.numel() for pm in model.parameters() if pm.requires_grad)
n_tot   = sum(pm.numel() for pm in model.parameters())
print(f"trainable params: {n_train:,} / {n_tot:,}")

opt = torch.optim.AdamW([pm for pm in model.parameters() if pm.requires_grad],
                        lr=args.lr, betas=(0.9, 0.95))

PM, PS = -1.2, 1.2
c_lo = 0.5 * (1 + math.erf((math.log(s_lo) - PM) / (PS * math.sqrt(2))))
c_hi = 0.5 * (1 + math.erf((math.log(s_hi) - PM) / (PS * math.sqrt(2))))

def sample_label_sigma(n):
	"""σ_label drawn from the target band's CDF range (matches training)."""
	u   = torch.rand(n)
	pcdf = c_lo + (c_hi - c_lo) * u
	ppf = math.sqrt(2.0) * torch.erfinv(2.0 * pcdf - 1.0)
	return torch.exp(PM + PS * ppf).to(device)

def make_sigmas(n, off_frac):
	sl = sample_label_sigma(n)
	is_off = torch.rand(n, device=device) < off_frac
	logm = torch.rand(n, device=device) * math.log(args.mismatch_max)
	m   = torch.where(is_off, torch.exp(logm), torch.ones(n, device=device))
	sn  = (sl * m).clamp(max=s_cap)
	return sn, sl

def rollout_edm_loss(X, Y):
	"""2b: EDM-weighted CE on ONE randomly-sampled realized drift state.

	Rollout is no_grad (cheap, detached); we backward through a single sampled
	state to keep activation memory at ~one 2S forward. Over steps this covers
	all depths stochastically — avoids retaining 8 graphs at once (OOM).
	"""
	states = model.block_rollout(X, TGT, args.rollout_steps, s_lo, s_hi)
	zt, sig = states[torch.randint(len(states), (1,)).item()]
	_, edm_loss, _ = model._forward_block_core(X, zt, Y, TGT,
	                                           sig.expand(X.size(0)))
	return edm_loss

@torch.no_grad()
def evaluate():
	"""on-shell CE (shell retention) + drift CE (the failure axis)."""
	model.eval()
	on, drift = [], []
	for _ in range(args.eval_iters):
		X, Y = get_batch('val')
		sl = sample_label_sigma(X.size(0))
		_, _, ce = model.forward_aug(X, Y, TGT, sl, sl)        # σ_noise=σ_label
		on.append(ce.item())
		if args.mode == 'sigma':
			sn, sl2 = make_sigmas(X.size(0), args.off_frac)
			_, _, ce = model.forward_aug(X, Y, TGT, sn, sl2)
			drift.append(ce.item())
		else:  # rollout: mean CE over realized drift states
			states = model.block_rollout(X, TGT, args.rollout_steps, s_lo, s_hi)
			ces = [model._forward_block_core(X, zt, Y, TGT,
			       sig.expand(X.size(0)))[2].item() for zt, sig in states]
			drift.append(float(np.mean(ces)))
	model.train()
	return float(np.mean(on)), float(np.mean(drift))

# --- fine-tune loop ---
# Failure axis = val DRIFT CE (block's own realized rollout states). Track its EMA
# over eval points: best ckpt = lowest EMA drift; early-stop when it flattens;
# warn on the shakespeare turn-up (train loss falling while EMA drift rises).
print(f"mode={args.mode}  ema_decay={args.ema_decay}  patience={args.patience}")
model.train()
curve, tr_ema = [], None
ema_drift, best_ema, best_iter, stale = None, 1e9, 0, 0
for it in range(args.max_iters + 1):
	if it % args.eval_interval == 0:
		on, drift = evaluate()
		ema_drift = drift if ema_drift is None else \
		            args.ema_decay * ema_drift + (1 - args.ema_decay) * drift
		improved = ema_drift < best_ema - args.min_delta
		turnup = (not improved) and (tr_ema is not None) and (best_iter > 0) \
		         and ema_drift > best_ema + args.min_delta
		flag = "  *best*" if improved else ("  ↑turn-up" if turnup else "")
		print(f"iter {it:5d}: on-shell CE {on:.4f}  drift CE {drift:.4f}  "
		      f"ema {ema_drift:.4f}  train {0.0 if tr_ema is None else tr_ema:.4f}{flag}",
		      flush=True)
		curve.append({'iter': it, 'on_shell_ce': on, 'drift_ce': drift,
		              'ema_drift': ema_drift,
		              'train_loss': None if tr_ema is None else tr_ema})
		with open(os.path.join(args.out_dir, 'loss_curves_2b.json'), 'w') as f:
			json.dump(curve, f, indent=1)
		if improved:
			best_ema, best_iter, stale = ema_drift, it, 0
			torch.save({'model': model.state_dict(), 'model_args': margs,
			            'iter_num': it, 'best_ema_drift': best_ema,
			            'config': vars(args)},
			           os.path.join(args.out_dir, 'ckpt.pt'))
		else:
			stale += 1
		torch.save({'model': model.state_dict(), 'model_args': margs,
		            'iter_num': it, 'config': vars(args)},
		           os.path.join(args.out_dir, 'last.pt'))
		if args.patience and stale >= args.patience:
			print(f"early-stop: EMA drift flat for {stale} evals "
			      f"(best {best_ema:.4f} @ iter {best_iter})")
			break
	if it == args.max_iters:
		break
	X, Y = get_batch('train')
	if args.mode == 'sigma':
		sn, sl = make_sigmas(X.size(0), args.off_frac)
		_, edm_loss, _ = model.forward_aug(X, Y, TGT, sn, sl)
	else:
		edm_loss = rollout_edm_loss(X, Y)
	opt.zero_grad(set_to_none=True)
	edm_loss.backward()
	opt.step()
	lv = edm_loss.item()
	tr_ema = lv if tr_ema is None else 0.98 * tr_ema + 0.02 * lv
	if args.log_interval and (it + 1) % args.log_interval == 0:
		print(f"  iter {it+1:5d}/{args.max_iters}  train {lv:.4f}  ema {tr_ema:.4f}",
		      flush=True)

print(f"done. best ckpt → {args.out_dir}/ckpt.pt  "
      f"(best EMA drift {best_ema:.4f} @ iter {best_iter}); last → last.pt")
