"""Training orchestration and controlled ablation experiments for support strategy models."""
import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import DataLoader

from zenova.schemas.strategy import SupportStrategy, DialogStage
from zenova.strategy.taxonomy import StrategyTaxonomy
from zenova.strategy.dataset import ESConvTurnSample
from zenova.strategy.baseline import StrategyTfidfBaseline
from zenova.strategy.transformer import (
    StrategyTransformerModel,
    StrategyDataset,
    StrategyTokenizer
)
from zenova.strategy.evaluator import StrategyEvaluator
from zenova.core.logging import get_logger

logger = get_logger("zenova.strategy.trainer")


class StrategyTransformerTrainer:
    """Trains and serializes PyTorch StrategyTransformerModel."""

    def __init__(
        self,
        taxonomy: Optional[StrategyTaxonomy] = None,
        max_vocab: int = 6000,
        max_len: int = 64,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 1,
        dim_feedforward: int = 128,
        dropout: float = 0.1
    ):
        self.taxonomy = taxonomy or StrategyTaxonomy.default()
        self.tokenizer = StrategyTokenizer(max_vocab=max_vocab, max_len=max_len)
        self.evaluator = StrategyEvaluator(taxonomy=self.taxonomy)
        self.max_len = max_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.model: Optional[StrategyTransformerModel] = None

    def prepare_data(
        self,
        samples: List[ESConvTurnSample],
        condition: str = "A"
    ) -> Tuple[List[str], List[int]]:
        """Format samples into input strings and integer label IDs."""
        texts = [s.format_input(condition=condition) for s in samples]
        labels = [self.taxonomy.label2id[s.target_strategy.value] for s in samples]
        return texts, labels

    def train(
        self,
        train_samples: List[ESConvTurnSample],
        val_samples: List[ESConvTurnSample],
        condition: str = "C",
        epochs: int = 4,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """Train transformer on designated condition and evaluate on validation set."""
        logger.info(f"Starting Strategy Transformer training (Condition {condition}, epochs={epochs}, lr={lr})...")
        train_texts, train_labels = self.prepare_data(train_samples, condition=condition)
        val_texts, val_labels = self.prepare_data(val_samples, condition=condition)

        # Build vocabulary from training texts
        self.tokenizer.build_vocab(train_texts)

        train_token_ids = [self.tokenizer.tokenize(t) for t in train_texts]
        val_token_ids = [self.tokenizer.tokenize(t) for t in val_texts]

        train_dataset = StrategyDataset(train_token_ids, train_labels, max_len=self.max_len)
        val_dataset = StrategyDataset(val_token_ids, val_labels, max_len=self.max_len)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        dev = torch.device(device)
        self.model = StrategyTransformerModel(
            vocab_size=len(self.tokenizer.vocab),
            num_classes=self.taxonomy.num_classes,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        ).to(dev)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        best_val_f1 = 0.0
        best_state = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            for batch_ids, batch_mask, batch_lbl in train_loader:
                batch_ids = batch_ids.to(dev)
                batch_mask = batch_mask.to(dev)
                batch_lbl = batch_lbl.to(dev)

                optimizer.zero_grad()
                logits = self.model(batch_ids, src_key_padding_mask=batch_mask)
                loss = criterion(logits, batch_lbl)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(train_loader))

            # Validation evaluation
            val_metrics = self.evaluate_loader(val_loader, dev)
            val_f1 = val_metrics["macro_f1"]
            val_acc = val_metrics["accuracy"]
            logger.info(
                f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.4f} - Val Macro-F1: {val_f1:.4f}"
            )

            if val_f1 >= best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(dev)

        final_val = self.evaluate_loader(val_loader, dev)
        return final_val

    def evaluate_loader(self, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
        """Evaluate model predictions on a DataLoader."""
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        self.model.eval()
        all_preds = []
        all_trues = []
        all_probs = []

        with torch.no_grad():
            for batch_ids, batch_mask, batch_lbl in loader:
                batch_ids = batch_ids.to(device)
                batch_mask = batch_mask.to(device)
                logits = self.model(batch_ids, src_key_padding_mask=batch_mask)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = logits.argmax(dim=-1).cpu().tolist()

                all_preds.extend([self.taxonomy.id2label[p] for p in preds])
                all_trues.extend([self.taxonomy.id2label[l.item()] for l in batch_lbl])

                for row in probs:
                    row_dict = {self.taxonomy.id2label[i]: float(row[i]) for i in range(len(row))}
                    all_probs.append(row_dict)

        return self.evaluator.evaluate(all_trues, all_preds, all_probs)

    def evaluate_samples(
        self,
        samples: List[ESConvTurnSample],
        condition: str = "C",
        batch_size: int = 64,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """Evaluate model on a list of samples under a specific condition."""
        texts, labels = self.prepare_data(samples, condition=condition)
        token_ids = [self.tokenizer.tokenize(t) for t in texts]
        ds = StrategyDataset(token_ids, labels, max_len=self.max_len)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        return self.evaluate_loader(loader, torch.device(device))

    def save(self, model_dir: str) -> None:
        """Serialize model checkpoint, tokenizer, and metadata."""
        p = Path(model_dir)
        p.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            torch.save({
                "state_dict": self.model.state_dict(),
                "vocab_size": len(self.tokenizer.vocab),
                "num_classes": self.taxonomy.num_classes,
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_layers": self.num_layers,
                "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout,
                "taxonomy": self.taxonomy.to_dict()
            }, p / "transformer_checkpoint.pt")

        self.tokenizer.save(str(p / "vocab.json"))

        metadata = {
            "model_type": "StrategyTransformerModel",
            "taxonomy_name": self.taxonomy.taxonomy_name,
            "num_classes": self.taxonomy.num_classes,
            "strategies": self.taxonomy.strategies,
            "max_len": self.max_len,
            "d_model": self.d_model
        }
        with open(p / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved Strategy Transformer checkpoint and artifacts to {p}")

    @classmethod
    def load(cls, model_dir: str, device: str = "cpu") -> "StrategyTransformerTrainer":
        """Load trained transformer and tokenizer from disk."""
        p = Path(model_dir)
        ckpt_path = p / "transformer_checkpoint.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        tax_data = ckpt.get("taxonomy")
        taxonomy = StrategyTaxonomy(
            taxonomy_name=tax_data.get("taxonomy_name", "ESConv-8"),
            strategies=tax_data.get("strategies"),
            descriptions=tax_data.get("descriptions")
        ) if tax_data else StrategyTaxonomy.default()

        trainer = cls(
            taxonomy=taxonomy,
            max_len=ckpt.get("max_len", 64),
            d_model=ckpt.get("d_model", 64),
            nhead=ckpt.get("nhead", 2),
            num_layers=ckpt.get("num_layers", 1),
            dim_feedforward=ckpt.get("dim_feedforward", 128),
            dropout=ckpt.get("dropout", 0.1)
        )
        trainer.tokenizer = StrategyTokenizer.load(str(p / "vocab.json"))
        trainer.model = StrategyTransformerModel(
            vocab_size=ckpt["vocab_size"],
            num_classes=ckpt["num_classes"],
            d_model=trainer.d_model,
            nhead=trainer.nhead,
            num_layers=trainer.num_layers,
            dim_feedforward=trainer.dim_feedforward,
            dropout=trainer.dropout
        )
        trainer.model.load_state_dict(ckpt["state_dict"])
        trainer.model.to(torch.device(device))
        trainer.model.eval()
        return trainer
