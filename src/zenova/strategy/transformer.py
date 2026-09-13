"""PyTorch Transformer sequence classifier for emotional support strategy detection."""
import re
import math
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from torch.utils.data import Dataset

from zenova.core.logging import get_logger

logger = get_logger("zenova.strategy.transformer")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class StrategyTransformerModel(nn.Module):
    """Multi-head self-attention sequence classifier built on torch.nn.TransformerEncoder."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pad_idx: int = 0
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)

        # Mean pooling across valid token positions
        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).unsqueeze(-1).float()
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = out.mean(dim=1)

        logits = self.classifier(pooled)
        return logits


class StrategyDataset(Dataset):
    """PyTorch Dataset for strategy token sequences and labels."""

    def __init__(self, token_ids_list: List[List[int]], labels: List[int], max_len: int = 128):
        self.data = token_ids_list
        self.labels = labels
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = self.data[idx][:self.max_len]
        length = len(ids)
        padded = ids + [0] * (self.max_len - length)
        mask = [False] * length + [True] * (self.max_len - length)
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


class StrategyTokenizer:
    """Tokenizer and vocabulary manager supporting conversational prefixes and special tokens."""

    def __init__(self, max_vocab: int = 8000, max_len: int = 128):
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.vocab: Dict[str, int] = {
            "<pad>": 0,
            "<unk>": 1,
            "<seeker>": 2,
            "<supporter>": 3,
            "<emotion>": 4,
            "<situation>": 5,
            "<problem>": 6,
        }

    def build_vocab(self, texts: List[str]) -> None:
        """Construct frequency-based vocabulary from training texts."""
        counts: Dict[str, int] = {}
        for t in texts:
            words = re.findall(r"\[.*?\]|\w+|[^\w\s]", t.lower())
            for w in words:
                counts[w] = counts.get(w, 0) + 1

        sorted_words = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for w, _ in sorted_words:
            if len(self.vocab) >= self.max_vocab:
                break
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab)
        logger.info(f"Built strategy vocabulary of size {len(self.vocab)}")

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text into integer token IDs."""
        words = re.findall(r"\[.*?\]|\w+|[^\w\s]", text.lower())
        return [self.vocab.get(w, self.vocab["<unk>"]) for w in words]

    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode single text into padded tensor and mask for inference."""
        ids = self.tokenize(text)[:self.max_len]
        length = len(ids)
        padded = ids + [0] * (self.max_len - length)
        mask = [False] * length + [True] * (self.max_len - length)
        return (
            torch.tensor([padded], dtype=torch.long),
            torch.tensor([mask], dtype=torch.bool)
        )

    def save(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab, "max_len": self.max_len}, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "StrategyTokenizer":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(max_len=data.get("max_len", 128))
        tok.vocab = data["vocab"]
        return tok
