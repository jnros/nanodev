"""DiffusionBlocks AR GPT (arxiv 2506.14202) on nanoGPT causal-LM.

EDM preconditioning + equi-probability block-σ curriculum.
All token positions noised with same σ; only the target block's
layers run per training step.
"""
import math
import random

import torch
import torch.nn as nn
from torch.nn import functional as F

from model import GPT, GPTConfig, LayerNorm, CausalSelfAttention, MLP


# ---------------------------------------------------------------------------
# σ schedule (no scipy -- uses torch.erfinv for probit)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
	return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
	t = torch.tensor(2.0 * p - 1.0, dtype=torch.float64)
	return float(math.sqrt(2.0) * torch.erfinv(t).item())


def get_block_sigmas(num_blocks: int, sigma_min: float = 0.002,
                     sigma_max: float = 80.0, p_mean: float = -1.2,
                     p_std: float = 1.2) -> list:
	# DBLOCK 1/6: partition log-normal σ distribution into B equal-probability
	# bands. each block owns one band; high σ = coarse, low σ = fine detail.
	"""Equi-probability σ partition boundaries; length = num_blocks + 1."""
	cdf_lo = _norm_cdf((math.log(sigma_min) - p_mean) / p_std)
	cdf_hi = _norm_cdf((math.log(sigma_max) - p_mean) / p_std)
	out = []
	for i in range(num_blocks + 1):
		p = cdf_lo + (cdf_hi - cdf_lo) * (i / num_blocks)
		out.append(math.exp(p_mean + p_std * _norm_ppf(p)))
	return out


# ---------------------------------------------------------------------------
# AdaLayerNorm
# ---------------------------------------------------------------------------

class AdaLayerNorm(nn.Module):
	# DBLOCK 2/6: noise-conditioned layer norm. injects c_noise (= 0.25·log σ)
	# as a per-sample scale+shift so each layer knows its noise level.
	"""LayerNorm + per-sample affine modulation from scalar c_noise."""

	def __init__(self, n_embd: int, bias: bool):
		super().__init__()
		self.ln = LayerNorm(n_embd, bias=bias)
		self.mod = nn.Sequential(
			nn.Linear(1, n_embd, bias=True),
			nn.SiLU(),
			nn.Linear(n_embd, 2 * n_embd, bias=True),
		)
		nn.init.zeros_(self.mod[-1].weight)
		nn.init.zeros_(self.mod[-1].bias)

	def forward(self, x: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
		# c_noise: [B]  →  mod: [B, 1, 2*n_embd]
		mod = self.mod(c_noise.unsqueeze(-1)).unsqueeze(1)
		scale, shift = mod.chunk(2, dim=-1)
		return (1.0 + scale) * self.ln(x) + shift


# ---------------------------------------------------------------------------
# DBlock: GPT block with AdaLayerNorm
# ---------------------------------------------------------------------------

class DBlock(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.ln_1 = AdaLayerNorm(config.n_embd, config.bias)
		self.attn = CausalSelfAttention(config)
		self.ln_2 = AdaLayerNorm(config.n_embd, config.bias)
		self.mlp  = MLP(config)

	def forward(self, x: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
		x = x + self.attn(self.ln_1(x, c_noise))
		x = x + self.mlp(self.ln_2(x, c_noise))
		return x


# ---------------------------------------------------------------------------
# GPTDBlock
# ---------------------------------------------------------------------------

class GPTDBlock(GPT):
	"""nanoGPT with DiffusionBlocks.

	B blocks partition the n_layer layers equally.  Each training step:
	  1. pick one block uniformly at random
	  2. sample σ from that block's CDF range
	  3. noise all token embeddings with that σ
	  4. run only that block's layers (AdaLN conditioned on c_noise)
	  5. EDM preconditioning on output
	  6. unembedding → logits → EDM-weighted CE loss
	"""

	def __init__(self, config: GPTConfig, num_dblocks: int = 3,
	             sigma_data: float = 0.5):
		assert config.n_layer % num_dblocks == 0, \
			f"n_layer {config.n_layer} not divisible by num_dblocks {num_dblocks}"
		super().__init__(config)
		self.num_dblocks = num_dblocks
		self.sigma_data  = sigma_data

		# DBLOCK 3/6: swap every standard Block for a DBlock (AdaLN variant).
		self.transformer.h = nn.ModuleList(
			[DBlock(config) for _ in range(config.n_layer)]
		)
		for blk in self.transformer.h:
			blk.apply(self._init_weights)
		for blk in self.transformer.h:
			for pn, p in blk.named_parameters():
				if pn.endswith('c_proj.weight'):
					torch.nn.init.normal_(
						p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

		self.block_sigmas = get_block_sigmas(num_dblocks)
		split = config.n_layer // num_dblocks
		self.layer_assignment = [
			list(range(i * split, (i + 1) * split))
			for i in range(num_dblocks)
		]

	# --- σ helpers ---

	def _sample_sigmas(self, n: int, device) -> torch.Tensor:
		"""n σ values drawn from a uniformly-chosen block's CDF range."""
		b     = random.randint(0, self.num_dblocks - 1)
		s_lo  = self.block_sigmas[b]
		s_hi  = self.block_sigmas[b + 1]
		pm, ps = -1.2, 1.2
		c_lo  = _norm_cdf((math.log(s_lo) - pm) / ps)
		c_hi  = _norm_cdf((math.log(s_hi) - pm) / ps)
		u     = torch.rand(n)
		p     = c_lo + (c_hi - c_lo) * u
		ppf   = math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)
		sigma = torch.exp(torch.tensor(pm, dtype=torch.float32)
		                  + torch.tensor(ps, dtype=torch.float32) * ppf)
		return sigma.to(device)

	def _target_block(self, sigma: torch.Tensor) -> int:
		"""Majority-vote block index; high σ → block 0 (first layers)."""
		bs   = torch.tensor(self.block_sigmas, device=sigma.device,
		                    dtype=sigma.dtype)
		idx  = torch.bucketize(sigma, bs, right=True) - 1
		idx  = (self.num_dblocks - 1) - idx
		idx  = idx.clamp(0, self.num_dblocks - 1).long()
		vals, counts = idx.unique(return_counts=True)
		return int(vals[counts.argmax()].item())

	def _edm_weights(self, sigma: torch.Tensor) -> torch.Tensor:
		w = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data)**2
		return w / w.mean()

	# --- forward ---

	def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
		device = idx.device
		b, t   = idx.size()
		assert t <= self.config.block_size

		# DBLOCK 4/6: sample σ from one block's CDF band (random block each step).
		sigma   = self._sample_sigmas(b, device)
		s2      = sigma ** 2
		sd2     = self.sigma_data ** 2
		c_skip  = sd2 / (s2 + sd2)
		c_out   = sigma * self.sigma_data / (s2 + sd2).sqrt()
		c_in    = 1.0 / (s2 + sd2).sqrt()
		c_noise = 0.25 * sigma.log()

		z   = self.transformer.wte(idx)
		# DBLOCK 5/6: noise all token embeddings with the sampled σ.
		zt  = z + sigma[:, None, None] * torch.randn_like(z)

		pos = torch.arange(t, device=device)
		x   = self.transformer.drop(
		        zt * c_in[:, None, None] + self.transformer.wpe(pos))

		# DBLOCK 6/6: run only the assigned block's layers — no gradient flows
		# to other blocks. this is the whole mechanism.
		blk = self._target_block(sigma)
		for i in self.layer_assignment[blk]:
			x = self.transformer.h[i](x, c_noise)

		x = self.transformer.ln_f(x)

		denoised = (x  * c_out[:, None, None]
		            + zt * c_skip[:, None, None])

		logits = self.lm_head(denoised)

		edm_loss = ce_loss = None
		if targets is not None:
			per_tok  = F.cross_entropy(
			    logits.view(-1, logits.size(-1)),
			    targets.view(-1),
			    reduction='none')
			ce_loss  = per_tok.mean()
			w        = self._edm_weights(sigma)
			per_seq  = per_tok.view(b, t).mean(-1)
			edm_loss = (per_seq * w).mean()

		return logits, edm_loss, ce_loss

	def forward_multistep_eval(self, idx: torch.Tensor, targets: torch.Tensor,
	                           n_steps: int = 1):
		"""ODE discretization diagnostic: Euler trajectory within each block's σ band.

		Each block evaluated independently (not chained). For each block:
		  - start from noisy token embeddings at σ_hi of that block's band
		  - take n_steps-1 Euler steps (prob-flow ODE) then x0-prediction at σ_lo
		  - total NFE per block = n_steps

		Returns:
		  ce_final  -- CE through lm_head for last block only (the only meaningful CE)
		  mse_list  -- MSE of denoised vs clean token embeddings, one per block

		Diagnostic: CE/MSE decreasing with n_steps → ODE discretization is real.
		Per-block MSE reveals whether gains concentrate in low-σ (fine) blocks.
		"""
		device = idx.device
		b, t = idx.size()
		z   = self.transformer.wte(idx)   # clean token embeddings [b,t,d]
		pos = torch.arange(t, device=device)

		def _denoise(zt, sigma_vec, blk_idx):
			# sigma_vec: [b] scalar σ per sample
			s2  = sigma_vec ** 2
			sd2 = self.sigma_data ** 2
			c_s = sd2 / (s2 + sd2)
			c_o = sigma_vec * self.sigma_data / (s2 + sd2).sqrt()
			c_i = 1.0 / (s2 + sd2).sqrt()
			c_n = 0.25 * sigma_vec.log()
			x = zt * c_i[:, None, None] + self.transformer.wpe(pos)
			for i in self.layer_assignment[blk_idx]:
				x = self.transformer.h[i](x, c_n)
			x = self.transformer.ln_f(x)
			return x * c_o[:, None, None] + zt * c_s[:, None, None]

		mse_list  = []
		ce_final  = None

		for blk in range(self.num_dblocks):
			# block 0 = high σ (coarse); reversed from block_sigmas (low→high)
			actual = self.num_dblocks - 1 - blk
			s_lo   = self.block_sigmas[actual]
			s_hi   = self.block_sigmas[actual + 1]

			# n_steps NFE: n_steps-1 Euler steps, then x0-prediction at sigmas[-1]
			# linspace(s_hi, s_lo, n_steps): n_steps=1 → [s_hi], n_steps=2 → [s_hi,s_lo]
			sigmas = torch.linspace(math.log(s_hi), math.log(s_lo),
			                        n_steps, device=device).exp()

			zt = z + sigmas[0] * torch.randn_like(z)

			for k in range(n_steps - 1):
				s_k   = sigmas[k].expand(b)
				s_kp1 = sigmas[k + 1].expand(b)
				denoised = _denoise(zt, s_k, blk)
				# probability-flow ODE Euler step (Karras et al. eq. 2)
				zt = zt + (zt - denoised) / s_k[:, None, None] \
				         * (s_kp1 - s_k)[:, None, None]

			# x0-prediction at sigmas[-1]: the n_steps-th (final) NFE
			denoised = _denoise(zt, sigmas[-1].expand(b), blk)

			mse = ((denoised - z) ** 2).mean()
			mse_list.append(mse)

			if blk == self.num_dblocks - 1:
				logits   = self.lm_head(denoised)
				ce_final = F.cross_entropy(
				    logits.view(-1, logits.size(-1)), targets.view(-1))

		return ce_final, mse_list
