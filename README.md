# DiffusionBlocks × nanoGPT

Independent language model port of Sakana's DiffusionBlocks (ICLR 2026). Applied to Andrej Karpathy's nanoGPT with Shakespeare-char training data.

![`summary.png`](summary.png)

Autoregressive language model. EDM preconditioning, σ sampled from the equi-probability log-normal partition, target embedding noised, only the assigned block's layers run, denoising prediction, logits out. 

Each block gets its own fresh noise sample at its assigned σ — not the output of the block before it. Each block has its own denoising objective. No backward gradient flows between blocks. Training moves from one monolith to a set of independent specialists.

The published Sakana repo implements only image classification; the AR language-model results in the paper aren't open-sourced. So I cloned Karpathy's nanoGPT (he's my sensei) and added diffusion where it don't belong.

6-layer causal GPT, Shakespeare-char, A100 GPU.

**Peak VRAM:** 1628 → 758 MB (2.16× reduction). This is the small win. Only 2 of 6 layers run per step on a single GPU. The bigger structural win: those 2 layers don't need the other 4 at all. They could be on a different machine, trained by a different team.

**Best val CE:** 1.46 (baseline) vs 2.05 (DiffusionBlocks). A real gap.

**Ablation:** hypothesized the gap was an EDM weighting artifact (low-σ blocks dominating gradients). Ran a normalized-weight ablation. Negative result. The gap is structural at this scale, not a weighting bug. 

## Code

Two files hold all the changes from nanoGPT:

- [`model_dblock.py`](model_dblock.py) — 205 lines. Imports the original `model.py` and adds the diffusion logic on top. The whole file is the delta. Six `DBLOCK n/6` markers walk you through the key moves in order.
- [`train_dblock.py`](train_dblock.py) — 318 lines. Training loop adapted for the diffusion objective: EDM-weighted loss, σ curriculum, dual logging of EDM loss and val CE.

Everything else is unmodified Karpathy.

---

Limitations: Not frontier scale. The paper's largest AR result is 12 layers on OpenWebText with mixed metrics. Open questions: Whether gap closes or widens with depth, fine-tuning works on pretrained models, and parallel-across-interconnect claim holds at realistic latencies. All testable.

If open questions resolve favorably, the entire concentration story behind the current AI buildout bends. That's a big if. It's also a smaller if than it was 48 hours ago.

![loss curves](loss_curves_overlay.png)
