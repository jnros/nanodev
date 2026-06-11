"""Plot CE vs sync_interval for the interconnect experiment.

Reads dblock_summary.json from out-interconnect-sync* dirs.
Reference lines from existing enwik8 L=6 single-device results.
"""
import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# single-device reference values (L=6, enwik8)
BASELINE_CE = 1.0151  # dense baseline, nats


def load_best_ce(dirpath):
	"""Return (sync_interval, best_val_ce) using loss_curves.json as authoritative source."""
	with open(os.path.join(dirpath, 'dblock_summary.json')) as f:
		s = json.load(f)
	with open(os.path.join(dirpath, 'loss_curves.json')) as f:
		curves = json.load(f)
	best = min(e['val_ce'] for e in curves)
	return s['sync_interval'], best


def main():
	# seed1337 = corrected K=∞ run; exclude old bogus out-interconnect-sync999999
	INF_DIR = 'out-interconnect-sync999999-seed1337'
	raw = glob.glob('out-interconnect-sync*')
	dirs = []
	for d in raw:
		suffix = d.split('sync')[-1]
		try:
			k = int(suffix)
		except ValueError:
			continue
		# substitute corrected dir for K=∞
		if k >= 999990 and os.path.exists(os.path.join(INF_DIR, 'dblock_summary.json')):
			if d == 'out-interconnect-sync999999':
				d = INF_DIR
		dirs.append((k, d))
	dirs = [d for _, d in sorted(dirs, key=lambda x: x[0])]
	if not dirs:
		print("no out-interconnect-sync* dirs found")
		return

	ks, ces = [], []
	for d in dirs:
		if not os.path.exists(os.path.join(d, 'loss_curves.json')):
			print(f"missing loss_curves.json in {d}, skipping")
			continue
		k, ce = load_best_ce(d)
		ks.append(k)
		ces.append(ce)
		print(f"sync_interval={k:>7d}  val_ce={ce:.4f}")

	fig, ax = plt.subplots(figsize=(7, 4))

	# split finite vs inf points
	fin_ks  = [k for k in ks  if k < 999999]
	fin_ces = [c for k, c in zip(ks, ces) if k < 999999]
	inf_ks  = [k for k in ks  if k >= 999999]
	inf_ces = [c for k, c in zip(ks, ces) if k >= 999999]

	ax.semilogx(fin_ks, fin_ces, 'o-', color='steelblue', linewidth=2,
	            markersize=7, label='interconnect (B=2, L=6)')
	if inf_ks:
		ax.semilogx(inf_ks, inf_ces, 'X', color='crimson',
		            markersize=12, markeredgewidth=2,
		            label=f'K=∞ (diverges)')
		ax.annotate('diverges', xy=(inf_ks[0], inf_ces[0]),
		            xytext=(-55, 12), textcoords='offset points',
		            fontsize=9, color='crimson',
		            arrowprops=dict(arrowstyle='->', color='crimson', lw=1.2))

	ax.axhline(BASELINE_CE, color='green', linestyle='--', linewidth=1.5,
	           label=f'dense baseline (L=6, CE={BASELINE_CE:.3f})')

	ax.set_xlabel('shared param sync interval (steps, log scale)')
	ax.set_ylabel('val CE (nats)')
	ax.set_title('Decoupling tolerance: CE vs sync frequency\n'
	             'enwik8, L=6, B=2, one block per GPU, PCIe-only')
	ax.legend(fontsize=9)
	ax.grid(True, which='both', alpha=0.3)
	plt.tight_layout()

	out = 'interconnect_sync_sweep.png'
	plt.savefig(out, dpi=150)
	print(f"saved {out}")


if __name__ == '__main__':
	main()
