"""Trainer for PyTorch Cost-Sensitive Risk Transformer Classifier."""
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from zenova.risk.transformer import RiskTransformerModel
from zenova.schemas.standard import RiskLevel
from zenova.core.logging import get_logger

logger = get_logger("zenova.risk.trainer")


class RiskClassificationDataset(Dataset):
    """PyTorch Dataset for multi-tier risk classification."""

    def __init__(
        self,
        texts: List[str],
        labels: List[str],
        vocab: Dict[str, int],
        label2id: Dict[str, int],
        max_len: int = 128,
        is_train: bool = False
    ):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.label2id = label2id
        self.max_len = max_len
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.texts)

    def tokenize(self, text: str) -> List[int]:
        tokens = re.findall(r"\w+", text.lower())
        ids = [self.vocab.get(t, 1) for t in tokens][:self.max_len]
        if self.is_train:
            import random
            ids = [1 if random.random() < 0.12 else token_id for token_id in ids]
        return ids

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = self.tokenize(self.texts[idx])
        length = len(ids)

        padded = ids + [0] * (self.max_len - length)
        mask = [False] * length + [True] * (self.max_len - length)

        label_id = self.label2id[self.labels[idx]]

        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(label_id, dtype=torch.long)
        )


class RiskTransformerTrainer:
    """Cost-sensitive trainer penalizing False Negatives on High/Critical crisis tiers."""

    def __init__(self, labels: Optional[List[str]] = None, max_vocab: int = 5000, max_len: int = 128):
        self.labels = labels or [
            RiskLevel.LOW.value,
            RiskLevel.MODERATE.value,
            RiskLevel.HIGH.value,
            RiskLevel.CRITICAL.value
        ]
        self.label2id = {lbl: i for i, lbl in enumerate(self.labels)}
        self.id2label = {i: lbl for i, lbl in enumerate(self.labels)}
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.vocab: Dict[str, int] = {}

    def build_vocab(self, texts: List[str]):
        counter = Counter()
        for t in texts:
            tokens = re.findall(r"\w+", t.lower())
            counter.update(tokens)

        vocab = {"<pad>": 0, "<unk>": 1}
        for word, _ in counter.most_common(self.max_vocab - 2):
            vocab[word] = len(vocab)
        self.vocab = vocab
        logger.info(f"Built risk vocabulary with {len(self.vocab)} tokens.")

    def train(
        self,
        train_texts: List[str],
        train_labels: List[str],
        val_texts: List[str],
        val_labels: List[str],
        epochs: int = 6,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = "cpu"
    ) -> RiskTransformerModel:
        if not self.vocab:
            self.build_vocab(train_texts)

        train_ds = RiskClassificationDataset(train_texts, train_labels, self.vocab, self.label2id, self.max_len, is_train=True)
        val_ds = RiskClassificationDataset(val_texts, val_labels, self.vocab, self.label2id, self.max_len, is_train=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = RiskTransformerModel(
            vocab_size=len(self.vocab),
            num_classes=len(self.labels),
            d_model=128,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            max_len=self.max_len
        ).to(device)

        # Asymmetric class weights: 1.0 for low, 1.2 for moderate, 2.0 for high, 2.5 for critical
        class_weights = torch.tensor([1.0, 1.2, 2.0, 2.5], dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

        logger.info(f"Starting Cost-Sensitive Risk Transformer training on {device} ({epochs} epochs)...")

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0

            for x_b, mask_b, y_b in train_loader:
                x_b = x_b.to(device)
                mask_b = mask_b.to(device)
                y_b = y_b.to(device)

                optimizer.zero_grad()
                logits = model(x_b, src_key_padding_mask=mask_b)
                loss = criterion(logits, y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total_val = 0
            val_preds = []
            val_trues = []

            with torch.no_grad():
                for x_val, mask_val, y_val in val_loader:
                    x_val = x_val.to(device)
                    mask_val = mask_val.to(device)
                    y_val = y_val.to(device)

                    logits = model(x_val, src_key_padding_mask=mask_val)
                    loss = criterion(logits, y_val)
                    val_loss += loss.item()

                    preds = torch.argmax(logits, dim=1)
                    val_preds.extend(preds.cpu().tolist())
                    val_trues.extend(y_val.cpu().tolist())
                    correct += (preds == y_val).sum().item()
                    total_val += y_val.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total_val if total_val > 0 else 0.0

            # Calculate Sensitivity on High/Critical (classes 2 and 3)
            high_crit_trues = [i for i, y in enumerate(val_trues) if y in (2, 3)]
            high_crit_detected = sum(1 for i in high_crit_trues if val_preds[i] in (2, 3))
            high_crit_recall = high_crit_detected / len(high_crit_trues) if high_crit_trues else 1.0

            logger.info(
                f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"High/Crit Sensitivity: {high_crit_recall:.4f}"
            )

        return model

    def save_artifacts(self, model: RiskTransformerModel, output_dir: str):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        meta = {
            "labels": self.labels,
            "label2id": self.label2id,
            "id2label": self.id2label,
            "vocab": self.vocab,
            "max_len": self.max_len,
            "d_model": model.d_model,
            "num_classes": model.num_classes,
            "version": "transformer-v1.0.0"
        }

        meta_path = out / "vocab_and_labels.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        ckpt_path = out / "transformer_checkpoint.pt"
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f"Saved Risk Transformer artifacts to {output_dir}")
