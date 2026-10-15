"""Official CFA text path: frozen BERT token embeddings + prompt wrap (no Transformer)."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

OFFICIAL_PROMPT = (
    "<|start_prompt|Make predictions about the future based on the following information: "
    "{text}<|<end_prompt>|>"
)


class BertEmbeddingPath(nn.Module):
    """Match Exp_Long_Term_Forecast with use_fullmodel=0.

    tokenizer → get_input_embeddings() only. BERT Transformer stays unused/frozen.
    """

    def __init__(
        self,
        model_name: str = "google-bert/bert-base-uncased",
        max_length: int = 1536,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.tokenizer.pad_token = "[PAD]"

        bert = BertModel.from_pretrained(model_name)
        self.embed = bert.get_input_embeddings()
        for p in self.embed.parameters():
            p.requires_grad = False
        self.embed.eval()
        self._device = device

    def to(self, device):  # type: ignore[override]
        self._device = torch.device(device) if not isinstance(device, torch.device) else device
        self.embed = self.embed.to(self._device)
        return self

    def wrap(self, texts: Sequence[str]) -> List[str]:
        return [OFFICIAL_PROMPT.format(text=t) for t in texts]

    @torch.no_grad()
    def encode_token_embeddings(self, texts: Sequence[str]) -> torch.Tensor:
        """Return [B, T, 768] token embeddings (no Transformer)."""
        if not texts:
            raise ValueError("texts must be non-empty")
        device = self._device or next(self.embed.parameters()).device
        wrapped = self.wrap(list(texts))
        ids = self.tokenizer(
            wrapped,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).input_ids.to(device)
        return self.embed(ids)

    def count_tokens(self, text: str, *, truncation: bool = False) -> int:
        wrapped = self.wrap([text])[0]
        ids = self.tokenizer(
            wrapped,
            truncation=truncation,
            max_length=self.max_length if truncation else None,
            add_special_tokens=True,
        ).input_ids
        return len(ids)
