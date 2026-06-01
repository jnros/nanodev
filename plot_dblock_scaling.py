"""DiffusionBlocks x nanoGPT — CE gap and VRAM vs depth.

Not part of upstream nanoGPT. Reads output dirs from the depth-scaling
experiment (L=6/8/10/12, 2 layers per dblock) and renders scaling_depth.png.
Run: uv run python plot_dblock_scaling.py
"""
import json
import os
import matplotlib.pyplot as plt

DEPTHS = [6, 8, 10, 12]

BASELINE_DIRS = {
	6:  'out-shakespeare-char-baseline',
	8:  'out-shakespeare-char-baseline-L8',
	10: 'out-shakespeare-char-baseline-L10',
	12: 'out-shakespeare-char-baseline-L12',
}

DBLOCK_DIRS = {
	6:  'out-shakespeare-char-dblock-B3',
	8:  'out-shakespeare-char-dblock-L8-B4',
	10: 'out-shakespeare-char-dblock-L10-B5',
	12: 'out-shakespeare-char-dblock-L12-B6',
}


def _final_val_ce(out_dir):
	path = os.path.join(out_dir, 'loss_curves.json')
	with open(path) as f:
		curves = json.load(f)
	return curves[-1].get('val_ce', curves[-1]['val'])


def _peak_vram(out_dir, kind):
	fname = 'baseline_summary.json' if kind == 'baseline' else 'dblock_summary.json'
	with open(os.path.join(out_dir, fname)) as f:
		return json.load(f)['peak_vram_mb']


def main():
	depths, base_ce, dblock_ce, gap, base_vram, dblock_vram = [], [], [], [], [], []

	for d in DEPTHS:
		bdir = BASELINE_DIRS[d]
		ddir = DBLOCK_DIRS[d]
		if not os.path.exists(os.path.join(bdir, 'loss_curves.json')):
			print(f"missing baseline L={d}, skipping")
			continue
		if not os.path.exists(os.path.join(ddir, 'loss_curves.json')):
			print(f"missing dblock L={d}, skipping")
			continue

		b = _final_val_ce(bdir)
		db = _final_val_ce(ddir)
		depths.append(d)
		base_ce.append(b)
		dblock_ce.append(db)
		gap.append(db - b)
		base_vram.append(_peak_vram(bdir, 'baseline'))
		dblock_vram.append(_peak_vram(ddir, 'dblock'))

	if not depths:
		print("no complete runs found")
		return

	fig, axes = plt.subplots(1, 3, figsize=(14, 4))

	ax = axes[0]
	ax.plot(depths, base_ce, 'o-', label='baseline')
	ax.plot(depths, dblock_ce, 's--', label='dblock (B=L/2)')
	ax.set_xlabel('n_layer')
	ax.set_ylabel('val CE')
	ax.set_title('Val CE vs depth')
	ax.legend()
	ax.grid(True, alpha=0.3)

	ax = axes[1]
	ax.plot(depths, gap, 'D-', color='tab:red')
	ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
	ax.set_xlabel('n_layer')
	ax.set_ylabel('dblock CE − baseline CE')
	ax.set_title('CE gap vs depth')
	ax.grid(True, alpha=0.3)

	ax = axes[2]
	ax.plot(depths, base_vram, 'o-', label='baseline')
	ax.plot(depths, dblock_vram, 's--', label='dblock')
	ax.set_xlabel('n_layer')
	ax.set_ylabel('peak VRAM (MB)')
	ax.set_title('Peak VRAM vs depth')
	ax.legend()
	ax.grid(True, alpha=0.3)

	fig.tight_layout()
	out = 'scaling_depth.png'
	fig.savefig(out, dpi=150)
	print(f"saved {out}")

	print(f"\n{'L':>4}  {'base CE':>8}  {'dblock CE':>10}  {'gap':>6}  {'base MB':>8}  {'dblock MB':>10}")
	for i, d in enumerate(depths):
		print(f"{d:>4}  {base_ce[i]:>8.4f}  {dblock_ce[i]:>10.4f}  "
		      f"{gap[i]:>+6.4f}  {base_vram[i]:>8.1f}  {dblock_vram[i]:>10.1f}")


if __name__ == '__main__':
	main()
