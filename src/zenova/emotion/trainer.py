"""Training and serialization pipeline for EmotionTransformerModel."""
import json
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

from zenova.emotion.transformer import EmotionTransformerModel
from zenova.core.logging import get_logger

logger = get_logger("zenova.emotion.trainer")


class EmotionDataset(Dataset):
    def __init__(self, token_ids_list: List[List[int]], labels: List[int], max_len: int = 64):
        self.data = token_ids_list
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx][:self.max_len]
        length = len(ids)
        padded = ids + [0] * (self.max_len - length)
        mask = [False] * length + [True] * (self.max_len - length)
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


class EmotionTransformerTrainer:
    """Trains and serializes PyTorch Transformer emotion classifier."""

    def __init__(self, labels: List[str], max_vocab: int = 6000, max_len: int = 64):
        self.labels = sorted(labels)
        self.label2id = {lbl: i for i, lbl in enumerate(self.labels)}
        self.id2label = {i: lbl for i, lbl in enumerate(self.labels)}
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.vocab: Dict[str, int] = {"<pad>": 0, "<unk>": 1}

    def build_vocab(self, texts: List[str]):
        counts = {}
        for t in texts:
            words = re.findall(r"\w+", t.lower())
            for w in words:
                counts[w] = counts.get(w, 0) + 1
        sorted_words = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for w, _ in sorted_words[:self.max_vocab - 2]:
            self.vocab[w] = len(self.vocab)
        logger.info(f"Built vocabulary of size {len(self.vocab)}")

    def tokenize(self, text: str) -> List[int]:
        words = re.findall(r"\w+", text.lower())
        return [self.vocab.get(w, 1) for w in words]

    def train(
        self,
        train_texts: List[str],
        train_labels: List[str],
        val_texts: List[str],
        val_labels: List[str],
        epochs: int = 5,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = "cpu"
    ) -> EmotionTransformerModel:
        self.build_vocab(train_texts)

        train_ids = [self.tokenize(t) for t in train_texts]
        train_y = [self.label2id[l] for l in train_labels]
        val_ids = [self.tokenize(t) for t in val_texts]
        val_y = [self.label2id[l] for l in val_labels]

        train_ds = EmotionDataset(train_ids, train_y, max_len=self.max_len)
        val_ds = EmotionDataset(val_ids, val_y, max_len=self.max_len)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = EmotionTransformerModel(
            vocab_size=len(self.vocab),
            num_classes=len(self.labels),
            d_model=128,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        best_val_loss = float("inf")
        best_state = None

        logger.info(f"Beginning Transformer training for {epochs} epochs on device: {device}...")
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for batch_x, batch_mask, batch_y in train_loader:
                batch_x, batch_mask, batch_y = batch_x.to(device), batch_mask.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = model(batch_x, src_key_padding_mask=batch_mask)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total_val = 0
            with torch.no_grad():
                for batch_x, batch_mask, batch_y in val_loader:
                    batch_x, batch_mask, batch_y = batch_x.to(device), batch_mask.to(device), batch_y.to(device)
                    logits = model(batch_x, src_key_padding_mask=batch_mask)
                    loss = criterion(logits, batch_y)
                    val_loss += loss.item()
                    preds = logits.argmax(dim=1)
                    correct += (preds == batch_y).sum().item()
                    total_val += batch_y.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total_val if total_val > 0 else 0.0

            logger.info(
                f"Epoch {epoch}/{epochs}: "
                f"Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Val Acc={val_acc:.4f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = model.state_dict()

        if best_state is not None:
            model.load_state_dict(best_state)

        return model

    def save_artifacts(self, model: EmotionTransformerModel, output_dir: str):
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), p / "transformer_checkpoint.pt")

        metadata = {
            "labels": self.labels,
            "label2id": self.label2id,
            "id2label": self.id2label,
            "vocab": self.vocab,
            "max_len": self.max_len,
            "d_model": model.d_model,
            "num_classes": len(self.labels)
        }
        with open(p / "vocab_and_labels.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        logger.info(f"Saved transformer model checkpoint and vocabulary to {output_dir}")
