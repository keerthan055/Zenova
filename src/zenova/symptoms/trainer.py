"""Trainer for PyTorch Multi-Label Symptom Transformer Classifier."""
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from zenova.symptoms.transformer import SymptomTransformerModel
from zenova.core.logging import get_logger

logger = get_logger("zenova.symptoms.trainer")


class MultiLabelSymptomDataset(Dataset):
    """PyTorch Dataset for multi-label sequence classification."""

    def __init__(
        self,
        texts: List[str],
        label_lists: List[List[str]],
        vocab: Dict[str, int],
        label2id: Dict[str, int],
        max_len: int = 128
    ):
        self.texts = texts
        self.label_lists = label_lists
        self.vocab = vocab
        self.label2id = label2id
        self.max_len = max_len
        self.num_classes = len(label2id)

    def __len__(self) -> int:
        return len(self.texts)

    def tokenize(self, text: str) -> List[int]:
        tokens = re.findall(r"\w+", text.lower())
        return [self.vocab.get(t, 1) for t in tokens][:self.max_len]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = self.tokenize(self.texts[idx])
        length = len(ids)

        padded = ids + [0] * (self.max_len - length)
        mask = [False] * length + [True] * (self.max_len - length)

        target = [0.0] * self.num_classes
        for lbl in self.label_lists[idx]:
            if lbl in self.label2id:
                target[self.label2id[lbl]] = 1.0

        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(target, dtype=torch.float)
        )


class SymptomTransformerTrainer:
    """End-to-end trainer for multi-label Transformer symptom identification."""

    def __init__(self, labels: List[str], max_vocab: int = 6000, max_len: int = 128):
        self.labels = sorted(labels)
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
        logger.info(f"Built vocabulary with {len(self.vocab)} tokens.")

    def train(
        self,
        train_texts: List[str],
        train_labels: List[List[str]],
        val_texts: List[str],
        val_labels: List[List[str]],
        epochs: int = 6,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = "cpu"
    ) -> SymptomTransformerModel:
        if not self.vocab:
            self.build_vocab(train_texts)

        train_ds = MultiLabelSymptomDataset(train_texts, train_labels, self.vocab, self.label2id, self.max_len)
        val_ds = MultiLabelSymptomDataset(val_texts, val_labels, self.vocab, self.label2id, self.max_len)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = SymptomTransformerModel(
            vocab_size=len(self.vocab),
            num_classes=len(self.labels),
            d_model=128,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            max_len=self.max_len
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

        logger.info(f"Starting Multi-Label Transformer training on {device} ({epochs} epochs, {len(train_loader)} batches/epoch)...")

        for epoch in range(1, epochs + 1):
            model.train()
            total_train_loss = 0.0

            for x_batch, mask_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                mask_batch = mask_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                logits = model(x_batch, src_key_padding_mask=mask_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation
            model.eval()
            total_val_loss = 0.0
            val_preds_list = []
            val_targets_list = []

            with torch.no_grad():
                for x_val, mask_val, y_val in val_loader:
                    x_val = x_val.to(device)
                    mask_val = mask_val.to(device)
                    y_val = y_val.to(device)

                    logits = model(x_val, src_key_padding_mask=mask_val)
                    loss = criterion(logits, y_val)
                    total_val_loss += loss.item()

                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_preds_list.append((probs >= 0.5).astype(int))
                    val_targets_list.append(y_val.cpu().numpy().astype(int))

            avg_val_loss = total_val_loss / len(val_loader)
            all_preds = np.vstack(val_preds_list)
            all_targets = np.vstack(val_targets_list)

            # Quick micro F1 check
            tp = np.sum((all_preds == 1) & (all_targets == 1))
            fp = np.sum((all_preds == 1) & (all_targets == 0))
            fn = np.sum((all_preds == 0) & (all_targets == 1))
            micro_f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)

            logger.info(
                f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Val Micro F1: {micro_f1:.4f}"
            )

        return model

    def save_artifacts(self, model: SymptomTransformerModel, output_dir: str):
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
        logger.info(f"Saved Transformer artifacts to {output_dir}")
