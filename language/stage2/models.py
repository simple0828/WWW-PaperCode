"""Shared SoftLSTM / Fact LatentCoT modules for Stage2."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftLSTMResid(nn.Module):
    def __init__(self, H: int, C: int, hidden: int = 128, n_token: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(C * 2, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.token_head = nn.Linear(hidden, n_token * hidden)
        self.n_token = n_token
        self.readout = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, H),
        )
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, ev):
        B, L, _, _ = ev.shape
        x = self.in_proj(ev.float().reshape(B, L, -1))
        out, _ = self.lstm(x)
        tokens = self.token_head(out[:, -1, :]).reshape(B, self.n_token, -1)
        return self.readout(tokens.mean(1))


class FactLatentCoT(nn.Module):
    """Fact LatentCoT residual head with optional Coconut / halt / mixture tricks.

    Tricks (orthogonal flags):
      1. coconut_recycle — feed last-layer latent hiddens back as next inputs
         (pure continuous-thought recycle; no residual blend with old lat).
      2. adaptive_halt — soft per-sample early exit weighted by fact density
         + halt head (ACT-style mass over steps).
      4. n_thought_groups>1 — split latents into parallel thought groups and
         fuse with attention (mixture / superposition).
    """

    def __init__(
        self,
        model_name: str,
        H: int,
        C: int,
        n_aux=4,
        n_latent=12,
        n_steps=3,
        max_length=256,
        use_aux: bool = True,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        learnable_amp: bool = False,
        amp_init: float = 0.2,
        coconut_recycle: bool = False,
        adaptive_halt: bool = False,
        n_thought_groups: int = 1,
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, trust_remote_code=True
        )
        for p in self.llm.parameters():
            p.requires_grad = False

        self.use_lora = bool(use_lora)
        if self.use_lora:
            from peft import LoraConfig, get_peft_model

            lora_cfg = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(self.llm, lora_cfg)

        # Needed so BS≈16–48 fits on ~32GB with long fact contexts (576–640).
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            self.llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if hasattr(self.llm, "enable_input_require_grads"):
            self.llm.enable_input_require_grads()

        hidden = int(self.llm.config.hidden_size)
        self.hidden = hidden
        self.C = C
        self.H = int(H)
        self.use_aux = bool(use_aux)
        self.n_aux = int(n_aux) if self.use_aux else 0
        self.n_latent = int(n_latent)
        self.n_steps = int(n_steps)
        self.max_length = max_length
        self.learnable_amp = bool(learnable_amp)
        self.coconut_recycle = bool(coconut_recycle)
        self.adaptive_halt = bool(adaptive_halt)
        self.n_thought_groups = max(1, int(n_thought_groups))
        if self.n_latent % self.n_thought_groups != 0:
            raise ValueError(
                f"n_latent={self.n_latent} must be divisible by "
                f"n_thought_groups={self.n_thought_groups}"
            )
        self.group_size = self.n_latent // self.n_thought_groups

        if self.use_aux:
            self.aux_proj = nn.Sequential(
                nn.Linear(C * 2, hidden), nn.GELU(), nn.Linear(hidden, self.n_aux * hidden)
            ).float()
        else:
            self.aux_proj = None
        self.latent_bias = nn.Parameter(torch.randn(self.n_latent, hidden) * 0.02)
        self.latent_from_ctx = nn.Linear(hidden, self.n_latent * hidden).float()
        # Legacy residual blend gate (used only when coconut_recycle=False).
        self.step_gate = nn.Parameter(torch.tensor(0.7))
        # Coconut: optional projection of recycled continuous thought.
        if self.coconut_recycle:
            self.thought_proj = nn.Linear(hidden, hidden, bias=False).float()
            nn.init.eye_(self.thought_proj.weight)
        else:
            self.thought_proj = None
        # Adaptive soft halt: p_exit from [ctx, density, step_emb, lat_mean].
        if self.adaptive_halt:
            self.halt_mlp = nn.Sequential(
                nn.Linear(hidden * 2 + 2, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
            ).float()
            self.step_embed = nn.Embedding(max(self.n_steps, 1), hidden).float()
        else:
            self.halt_mlp = None
            self.step_embed = None
        # Mixture-of-thoughts fusion over groups.
        if self.n_thought_groups > 1:
            self.group_query = nn.Linear(hidden, hidden, bias=False).float()
            self.group_key = nn.Linear(hidden, hidden, bias=False).float()
        else:
            self.group_query = None
            self.group_key = None

        self.readout = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, H),
        ).float()
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        # Trainable amplitude ∈ (0,1); init near amp_init via logit.
        if self.learnable_amp:
            a0 = float(np.clip(amp_init, 1e-3, 1 - 1e-3))
            logit0 = math.log(a0 / (1.0 - a0))
            self.amp_logit = nn.Parameter(torch.tensor(logit0, dtype=torch.float32))
        else:
            self.amp_logit = None

    def trainable_param_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": int(total), "trainable": int(train)}

    def _embed(self):
        base = self.llm.get_base_model() if hasattr(self.llm, "get_base_model") else self.llm
        if hasattr(base, "model") and hasattr(base.model, "embed_tokens"):
            return base.model.embed_tokens
        return base.get_input_embeddings()

    def _fuse_latents(self, lat: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """[B,n_latent,H] → [B,H]; mixture attention if n_thought_groups>1."""
        if self.n_thought_groups <= 1:
            return lat.float().mean(1)
        B = lat.size(0)
        groups = lat.float().reshape(B, self.n_thought_groups, self.group_size, self.hidden)
        g = groups.mean(2)  # [B,K,H]
        q = self.group_query(ctx.float()).unsqueeze(1)  # [B,1,H]
        k = self.group_key(g)  # [B,K,H]
        score = (q * k).sum(-1) / math.sqrt(float(self.hidden))  # [B,K]
        alpha = torch.softmax(score, dim=-1).unsqueeze(-1)  # [B,K,1]
        return (alpha * g).sum(1)

    def forward(self, input_ids, attention_mask, ev):
        B = input_ids.size(0)
        C = self.C
        dtype = next(self.llm.parameters()).dtype
        text_emb = self._embed()(input_ids).to(dtype=dtype)
        mask_f = attention_mask.unsqueeze(-1).to(dtype=dtype)
        dens = attention_mask.float().sum(1, keepdim=True) / float(
            max(attention_mask.size(1), 1)
        )  # [B,1] fact-length proxy
        ctx = (text_emb * mask_f).sum(1) / mask_f.sum(1).clamp_min(1.0)
        lat = (
            self.latent_from_ctx(ctx.float())
            .reshape(B, self.n_latent, self.hidden)
            .to(dtype=dtype)
        )
        lat = lat + self.latent_bias.unsqueeze(0).to(dtype=dtype)
        gate = torch.sigmoid(self.step_gate).to(dtype=dtype)

        if self.use_aux:
            flat = ev.float().reshape(B, ev.size(1), -1).mean(1)
            recent = ev.float()[:, -3:].reshape(B, -1, C * 2).mean(1)
            aux = (
                self.aux_proj(0.5 * flat + 0.5 * recent)
                .reshape(B, self.n_aux, self.hidden)
                .to(dtype=dtype)
            )
            extra = aux
            n_extra = self.n_aux + self.n_latent
        else:
            extra = None
            n_extra = self.n_latent

        # Soft halt accumulators (ACT-style).
        if self.adaptive_halt:
            remain = torch.ones(B, 1, device=input_ids.device, dtype=torch.float32)
            weighted = torch.zeros(B, self.H, device=input_ids.device, dtype=torch.float32)
            halt_eps = 1e-3
        else:
            remain = weighted = None
            halt_eps = None

        for t in range(self.n_steps):
            if extra is not None:
                embeds = torch.cat([text_emb, extra, lat], dim=1)
            else:
                embeds = torch.cat([text_emb, lat], dim=1)
            attn = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        B,
                        n_extra,
                        device=input_ids.device,
                        dtype=attention_mask.dtype,
                    ),
                ],
                dim=1,
            )
            out = self.llm(
                inputs_embeds=embeds,
                attention_mask=attn,
                output_hidden_states=True,
                use_cache=False,
            )
            new_lat = out.hidden_states[-1][:, -self.n_latent :, :]
            if self.coconut_recycle:
                # Continuous thought: recycled hiddens become next input embeds.
                lat = self.thought_proj(new_lat.float()).to(dtype=dtype)
            else:
                lat = gate * new_lat + (1 - gate) * lat

            if self.adaptive_halt:
                lat_mean = lat.float().mean(1)
                step_frac = torch.full(
                    (B, 1),
                    float(t) / float(max(self.n_steps - 1, 1)),
                    device=input_ids.device,
                    dtype=torch.float32,
                )
                # Optional step embedding mixed into lat_mean for richer halt features.
                step_e = self.step_embed(
                    torch.full(
                        (B,),
                        min(t, self.step_embed.num_embeddings - 1),
                        device=input_ids.device,
                        dtype=torch.long,
                    )
                )
                halt_in = torch.cat(
                    [ctx.float(), lat_mean + 0.1 * step_e, dens, step_frac],
                    dim=-1,
                )  # [B, 2H+2]
                p_exit = torch.sigmoid(self.halt_mlp(halt_in))  # [B,1]
                # Density prior: sparse facts → prefer earlier exit.
                p_exit = (p_exit * (0.35 + 0.65 * dens)).clamp(halt_eps, 1.0 - halt_eps)
                if t == self.n_steps - 1:
                    p_exit = torch.ones_like(p_exit)
                w = remain * p_exit
                fused = self._fuse_latents(lat, ctx)
                step_pred = self.readout(fused)
                weighted = weighted + w * step_pred
                remain = remain * (1.0 - p_exit)

        if self.adaptive_halt:
            raw = weighted
        else:
            fused = self._fuse_latents(lat, ctx)
            raw = self.readout(fused)
        if self.amp_logit is not None:
            raw = raw * torch.sigmoid(self.amp_logit)
        return raw

    def amp_value(self) -> float:
        if self.amp_logit is None:
            return 1.0
        return float(torch.sigmoid(self.amp_logit.detach()).item())
