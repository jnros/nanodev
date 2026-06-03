"""Plot CE vs sync_interval for the interconnect experiment.

Reads dblock_summary.json from out-interconnect-sync* dirs.
Reference lines from existing enwik8 L=6 single-device results.
"""
import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# single-device reference values from ce_gap_vs_depth.csv (L=6)
BASELINE_CE = 1.4704
DBLOCK_CE   = 2.0753  # single-device B=3, 90k iters


def load_summary(path):
	with open(path) as f:
		return json.load(f)


def main():
	dirs = sorted(glob.glob('out-interconnect-sync*'),
	              key=lambda d: int(d.split('sync')[-1]))
	if not dirs:
		print("no out-interconnect-sync* dirs found")
		return

	ks, ces = [], []
	for d in dirs:
		p = os.path.join(d, 'dblock_summary.json')
		if not os.path.exists(p):
			print(f"missing {p}, skipping")
			continue
		s = load_summary(p)
		k = s['sync_interval']
		# treat 999999 as "never" for display
		ks.append(k)
		ces.append(s['best_val_ce'])
		print(f"sync_interval={k:>7d}  val_ce={s['best_val_ce']:.4f}")

	fig, ax = plt.subplots(figsize=(7, 4))

	ax.semilogx(ks, ces, 'o-', color='steelblue', linewidth=2,
	            markersize=7, label='interconnect (B=2, L=6)')

	ax.axhline(BASELINE_CE, color='green', linestyle='--', linewidth=1.5,
	           label=f'dense baseline (L=6, CE={BASELINE_CE:.3f})')
	ax.axhline(DBLOCK_CE, color='orange', linestyle='--', linewidth=1.5,
	           label=f'single-device dblock B=3 (CE={DBLOCK_CE:.3f})')

	# label "never sync" point
	if ks and ks[-1] >= 999999:
		ax.annotate('never sync', xy=(ks[-1], ces[-1]),
		            xytext=(-60, 10), textcoords='offset points',
		            fontsize=9, color='steelblue')

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
