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
		# c_noise: [B] (broadcast over T)  ->  mod [B,1,2d]
		#       or [B,T] (per-position, clean-noisy 2S)  ->  mod [B,T,2d]
		if c_noise.dim() == 1:
			mod = self.mod(c_noise.unsqueeze(-1)).unsqueeze(1)
		else:
			mod = self.mod(c_noise.unsqueeze(-1))
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

	def forward(self, x: torch.Tensor, c_noise: torch.Tensor,
	            attn_mask=None) -> torch.Tensor:
		x = x + self.attn(self.ln_1(x, c_noise), attn_mask=attn_mask)
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
		self._mask_cache = {}

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

	def _clean_noisy_mask(self, s: int, device) -> torch.Tensor:
		"""2S×2S bool mask (True = attend) for [clean(0..s-1), noisy(0..s-1)].

		  clean→clean : causal      (k ≤ q)        — clean memory self-builds
		  clean→noisy : blocked                    — clean never sees noisy
		  noisy→clean : strict causal (k < q)      — denoise on REAL (clean) past
		  noisy→noisy : diagonal only (k == q)     — no noisy-to-noisy leakage
		"""
		key = (s, str(device))
		m = self._mask_cache.get(key)
		if m is not None:
			return m
		i = torch.arange(s, device=device)
		causal = i[:, None] >= i[None, :]              # [s,s] k ≤ q
		strict = i[:, None] >  i[None, :]              # [s,s] k < q
		eye    = torch.eye(s, dtype=torch.bool, device=device)
		zero   = torch.zeros(s, s, dtype=torch.bool, device=device)
		top    = torch.cat([causal, zero], dim=1)      # clean queries
		bot    = torch.cat([strict, eye],  dim=1)      # noisy queries
		m = torch.cat([top, bot], dim=0)               # [2s,2s]
		self._mask_cache[key] = m
		return m

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

		# clean stream conditioning: σ_min sentinel = bottom of lowest block band.
		sigma_min     = self.block_sigmas[0]
		c_in_clean    = 1.0 / math.sqrt(sigma_min ** 2 + sd2)
		c_noise_clean = 0.25 * math.log(sigma_min)

		z   = self.transformer.wte(idx)                       # clean  [b,t,d]
		# DBLOCK 5/6: noise token embeddings (noisy stream only).
		zt  = z + sigma[:, None, None] * torch.randn_like(z)  # noisy  [b,t,d]

		# Block-Diffusion clean-conditioning: concat clean+noisy → 2t.
		# both halves share positions 0..t-1.
		pos = torch.arange(t, device=device)
		wpe = self.transformer.wpe(pos)                       # [t,d]
		x_clean = z  * c_in_clean        + wpe
		x_noisy = zt * c_in[:, None, None] + wpe
		x = self.transformer.drop(torch.cat([x_clean, x_noisy], dim=1))  # [b,2t,d]

		# per-position c_noise: clean half = σ_min sentinel, noisy half = sampled σ.
		cn = torch.cat([
		        torch.full((b, t), c_noise_clean, device=device,
		                   dtype=c_noise.dtype),
		        c_noise[:, None].expand(b, t),
		     ], dim=1)                                        # [b,2t]

		mask = self._clean_noisy_mask(t, device)              # [2t,2t]

		# DBLOCK 6/6: run only the assigned block's layers — no gradient flows
		# to other blocks. this is the whole mechanism.
		blk = self._target_block(sigma)
		for i in self.layer_assignment[blk]:
			x = self.transformer.h[i](x, cn, attn_mask=mask)

		x = self.transformer.ln_f(x)

		x_noisy_out = x[:, t:, :]                             # noisy half only
		denoised = (x_noisy_out * c_out[:, None, None]
		            + zt        * c_skip[:, None, None])

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

	# --- chained (composed) inference: the deployment sampler ---

	def _build_sigma_grid(self, n_steps: int, grid: str, device,
	                      sigma_min=None, sigma_max=None) -> torch.Tensor:
		"""Descending σ schedule spanning the full range (high → low)."""
		s_min = sigma_min if sigma_min is not None else self.block_sigmas[0]
		s_max = sigma_max if sigma_max is not None else self.block_sigmas[-1]

		if grid == 'global':
			# one geometric (log-uniform) grid over whole range, descending
			return torch.linspace(math.log(s_max), math.log(s_min),
			                      n_steps, device=device).exp()

		if grid == 'perband':
			# n_steps points inside EACH block's band, top band → bottom band.
			# block 0 owns the highest band; block_sigmas ascending, so reverse.
			chunks = []
			for blk in range(self.num_dblocks):
				actual = self.num_dblocks - 1 - blk
				lo = self.block_sigmas[actual]
				hi = self.block_sigmas[actual + 1]
				band = torch.linspace(math.log(hi), math.log(lo),
				                      n_steps, device=device).exp()
				# drop duplicated boundary between consecutive bands
				chunks.append(band if blk == 0 else band[1:])
			return torch.cat(chunks)

		raise ValueError(f"unknown grid '{grid}'")

	def _denoise_chained(self, z_clean, zt, sigma_scalar, pos):
		"""Clean-conditioned denoiser query at one scalar σ; block chosen by σ.

		Block-Diffusion clean-conditioning: the noisy stream `zt` is denoised
		conditioned on the FIXED clean stream `z_clean` (clean past only, via the
		2S mask) — the AR property the old single-stream path lacked. Only the
		noisy half is returned. x0 = softmax(logits) @ wte.weight is taken by the
		caller (Sakana diffusion_step semantics; weight-tied head).

		Returns (logits, block_idx) for the noisy half [b,t,V].
		"""
		b, t = zt.size(0), zt.size(1)
		sig = sigma_scalar.expand(b)
		s2  = sig ** 2
		sd2 = self.sigma_data ** 2
		c_skip  = sd2 / (s2 + sd2)
		c_out   = sig * self.sigma_data / (s2 + sd2).sqrt()
		c_in    = 1.0 / (s2 + sd2).sqrt()
		c_noise = 0.25 * sig.log()

		sigma_min     = self.block_sigmas[0]
		c_in_clean    = 1.0 / math.sqrt(sigma_min ** 2 + sd2)
		c_noise_clean = 0.25 * math.log(sigma_min)

		wpe = self.transformer.wpe(pos)
		x_clean = z_clean * c_in_clean        + wpe
		x_noisy = zt      * c_in[:, None, None] + wpe
		x = self.transformer.drop(torch.cat([x_clean, x_noisy], dim=1))

		cn = torch.cat([
		        torch.full((b, t), c_noise_clean, device=zt.device,
		                   dtype=c_noise.dtype),
		        c_noise[:, None].expand(b, t),
		     ], dim=1)
		mask = self._clean_noisy_mask(t, zt.device)

		blk = self._target_block(sig)                       # the block switch
		for i in self.layer_assignment[blk]:
			x = self.transformer.h[i](x, cn, attn_mask=mask)
		x = self.transformer.ln_f(x)

		x_noisy_out = x[:, t:, :]
		denoised = x_noisy_out * c_out[:, None, None] + zt * c_skip[:, None, None]
		logits   = self.lm_head(denoised)
		return logits, blk

	@torch.no_grad()
	def forward_chained_eval(self, idx: torch.Tensor, targets: torch.Tensor,
	                         n_steps: int = 64, solver: str = 'euler',
	                         grid: str = 'global', from_noise: bool = True,
	                         sigma_min=None, sigma_max=None):
		"""Composed trajectory across ALL blocks — TEACHER-FORCED composed sampler.

		Walks the full descending σ schedule, switching blocks via _target_block
		as σ descends, threading ODE state z across block boundaries.

		NOT the free-running deployment sampler. The clean conditioning stream is
		the GROUND-TRUTH tokens (teacher forcing), not the model's own previously
		generated tokens. So this measures the composed-sampler likelihood given a
		correct clean context — an honest, useful instrument for the composed
		objective, but it does NOT expose error accumulation from the model
		conditioning on its own (possibly wrong) generations. Free-running
		generation is a separate path (see chained_eval.py notes).

		CLEAN-CONDITIONING (Block Diffusion): the evolving noisy stream is denoised
		conditioned on the fixed clean stream via the 2S mask in _denoise_chained —
		noisy reads clean PAST only. The AR property the old single-stream path lacked.

		x0 = softmax(logits) @ wte.weight  (Sakana diffusion_step; weight-tied head).
		Euler prob-flow ODE in embedding space; optional Heun corrector.

		from_noise=True  : pure-noise init of the noisy stream (clean past still
		                   supplied as teacher-forced conditioning).
		from_noise=False : z_clean + σ_max·noise init (reconstruction framing).

		Returns dict:
		  ce            -- CE through lm_head at the final (lowest) σ vs targets
		  mse_to_clean  -- ||final x0 − clean embeddings||²  (drift indicator)
		  blocks        -- block idx visited per step (shows the chaining)
		  mse_trace     -- per-step ||x0_k − clean||²  (watch for compounding)
		"""
		device   = idx.device
		b, t     = idx.size()
		pos      = torch.arange(t, device=device)
		z_clean  = self.transformer.wte(idx)                # [b,t,d]
		wte_w    = self.transformer.wte.weight              # [V,d] (tied head)

		sigmas = self._build_sigma_grid(n_steps, grid, device,
		                                sigma_min, sigma_max)

		if from_noise:
			zt = torch.randn_like(z_clean) * math.sqrt(1.0 + sigmas[0].item()**2)
		else:
			zt = z_clean + sigmas[0] * torch.randn_like(z_clean)

		blocks, mse_trace = [], []

		for k in range(sigmas.shape[0] - 1):
			s_k   = sigmas[k]
			s_kp1 = sigmas[k + 1]
			dt    = s_kp1 - s_k

			logits, blk = self._denoise_chained(z_clean, zt, s_k, pos)
			blocks.append(blk)
			x0 = F.softmax(logits, dim=-1) @ wte_w
			mse_trace.append(((x0 - z_clean) ** 2).mean().item())

			d = (zt - x0) / s_k

			if solver == 'heun':
				zt_pred = zt + d * dt
				logits2, _ = self._denoise_chained(z_clean, zt_pred, s_kp1, pos)
				x0_2 = F.softmax(logits2, dim=-1) @ wte_w
				d2   = (zt_pred - x0_2) / s_kp1
				zt   = zt + 0.5 * (d + d2) * dt
			else:  # euler
				zt = zt + d * dt

		# final denoise at lowest σ — the CE we report (Sakana returns
		# denoise(...) at min σ).
		logits, blk = self._denoise_chained(z_clean, zt, sigmas[-1], pos)
		blocks.append(blk)
		x0_final = F.softmax(logits, dim=-1) @ wte_w

		ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

		return {
			'ce': ce.item(),
			'mse_to_clean': ((x0_final - z_clean) ** 2).mean().item(),
			'blocks': blocks,
			'mse_trace': mse_trace,
		}

	def forward_multistep_eval(self, idx: torch.Tensor, targets: torch.Tensor,
	                           n_steps: int = 1, solver: str = 'euler'):
		"""ODE discretization diagnostic: ODE trajectory within each block's σ band.

		Each block evaluated independently (not chained). For each block:
		  - start from noisy token embeddings at σ_hi of that block's band
		  - take n_steps-1 solver steps (prob-flow ODE) then x0-prediction at σ_lo

		solver='euler': n_steps NFE per block (Karras et al. eq. 2)
		solver='heun':  2*(n_steps-1)+1 NFE per block (EDM Algorithm 1)

		Returns:
		  ce_final  -- CE through lm_head for last block only (the only meaningful CE)
		  mse_list  -- MSE of denoised vs clean token embeddings, one per block
		"""
		device = idx.device
		b, t = idx.size()
		z   = self.transformer.wte(idx)   # clean token embeddings [b,t,d]
		pos = torch.arange(t, device=device)

		def _denoise(zt, sigma_vec, blk_idx):
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

			# linspace in log-σ: n_steps=1 → [s_hi], n_steps=2 → [s_hi, s_lo]
			sigmas = torch.linspace(math.log(s_hi), math.log(s_lo),
			                        n_steps, device=device).exp()

			zt = z + sigmas[0] * torch.randn_like(z)

			for k in range(n_steps - 1):
				s_k   = sigmas[k].expand(b)
				s_kp1 = sigmas[k + 1].expand(b)
				dt    = (s_kp1 - s_k)[:, None, None]
				denoised_k = _denoise(zt, s_k, blk)
				d1 = (zt - denoised_k) / s_k[:, None, None]
				if solver == 'heun':
					# 2nd-order Heun corrector (EDM Algorithm 1)
					zt_pred    = zt + d1 * dt
					denoised_p = _denoise(zt_pred, s_kp1, blk)
					d2 = (zt_pred - denoised_p) / s_kp1[:, None, None]
					zt = zt + 0.5 * (d1 + d2) * dt
				elif solver == 'renoise':
					# ancestral/SDE: re-noise x0 estimate — forces every query
					# back to training distribution (z + σ'·noise)
					zt = denoised_k + s_kp1[:, None, None] * torch.randn_like(z)
				else:
					zt = zt + d1 * dt

			# final x0-prediction (1 NFE)
			denoised = _denoise(zt, sigmas[-1].expand(b), blk)

			mse = ((denoised - z) ** 2).mean()
			mse_list.append(mse)

			if blk == self.num_dblocks - 1:
				logits   = self.lm_head(denoised)
				ce_final = F.cross_entropy(
				    logits.view(-1, logits.size(-1)), targets.view(-1))

		return ce_final, mse_list
