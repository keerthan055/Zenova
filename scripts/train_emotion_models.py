import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import torch
from zenova.emotion.baseline import EmotionTfidfBaseline
from zenova.emotion.trainer import EmotionTransformerTrainer
from zenova.emotion.evaluator import EmotionModelEvaluator
from zenova.emotion.inference import EmotionInferenceEngine
from zenova.core.logging import get_logger

logger = get_logger("zenova.scripts.train_emotion")


def load_jsonl(filepath: str):
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def main():
    processed_dir = Path("data/processed/goemotions_ekman")
    train_file = processed_dir / "train.jsonl"
    val_file = processed_dir / "val.jsonl"
    test_file = processed_dir / "test.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(f"Missing train split at {train_file}. Run scripts/ingest_goemotions.py first.")

    train_data = load_jsonl(str(train_file))
    val_data = load_jsonl(str(val_file))
    test_data = load_jsonl(str(test_file))

    train_texts = [s["text"] for s in train_data]
    train_labels = [s["label"] for s in train_data]
    val_texts = [s["text"] for s in val_data]
    val_labels = [s["label"] for s in val_data]
    test_texts = [s["text"] for s in test_data]
    test_labels = [s["label"] for s in test_data]

    labels = sorted(list(set(train_labels)))
    logger.info(f"Loaded splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)} across {len(labels)} labels: {labels}")

    output_dir = "models/emotion"

    # =========================================================================
    # 1. Train and Evaluate Baseline (TF-IDF + Classifier)
    # =========================================================================
    print("\n--- 1. Training TF-IDF Baseline Classifier ---")
    baseline = EmotionTfidfBaseline(labels=labels)
    baseline.train(train_texts, train_labels)
    baseline.save(output_dir)

    baseline_preds = baseline.predict(test_texts)
    baseline_eval = EmotionModelEvaluator.evaluate(test_labels, baseline_preds, labels=labels)
    print(f"Baseline Accuracy: {baseline_eval['accuracy']:.4f}")
    print(f"Baseline Macro F1: {baseline_eval['macro_f1']:.4f}")
    print("Baseline Per-Class F1:")
    for lbl, m in baseline_eval["per_class"].items():
        print(f"  • {lbl:10s}: F1={m['f1']:.4f} (P={m['precision']:.4f}, R={m['recall']:.4f}, n={m['support']})")

    # =========================================================================
    # 2. Train and Evaluate PyTorch Transformer Sequence Classifier
    # =========================================================================
    print("\n--- 2. Training PyTorch Transformer Sequence Classifier ---")
    trainer = EmotionTransformerTrainer(labels=labels, max_vocab=6000, max_len=64)
    transformer_model = trainer.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        epochs=5,
        batch_size=64,
        lr=1e-3,
        device="cpu"
    )
    trainer.save_artifacts(transformer_model, output_dir)

    # Evaluate Transformer via Inference Engine on Test Split
    inference_engine = EmotionInferenceEngine(model_dir=output_dir)
    transformer_preds = [inference_engine.predict(t)["primary_emotion"] for t in test_texts]
    transformer_eval = EmotionModelEvaluator.evaluate(test_labels, transformer_preds, labels=labels)

    print(f"\nTransformer Accuracy: {transformer_eval['accuracy']:.4f}")
    print(f"Transformer Macro F1: {transformer_eval['macro_f1']:.4f}")
    print("Transformer Per-Class F1:")
    for lbl, m in transformer_eval["per_class"].items():
        print(f"  • {lbl:10s}: F1={m['f1']:.4f} (P={m['precision']:.4f}, R={m['recall']:.4f}, n={m['support']})")

    # =========================================================================
    # 3. Save Comparative Evaluation Report
    # =========================================================================
    report = {
        "dataset": "GoEmotions Ekman",
        "splits": {
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data)
        },
        "labels": labels,
        "baseline_model": {
            "name": "TF-IDF + LogisticRegression",
            "accuracy": baseline_eval["accuracy"],
            "macro_precision": baseline_eval["macro_precision"],
            "macro_recall": baseline_eval["macro_recall"],
            "macro_f1": baseline_eval["macro_f1"],
            "per_class": baseline_eval["per_class"],
            "confusion_matrix": baseline_eval["confusion_matrix"]
        },
        "transformer_model": {
            "name": "PyTorch TransformerEncoder Classifier",
            "accuracy": transformer_eval["accuracy"],
            "macro_precision": transformer_eval["macro_precision"],
            "macro_recall": transformer_eval["macro_recall"],
            "macro_f1": transformer_eval["macro_f1"],
            "per_class": transformer_eval["per_class"],
            "confusion_matrix": transformer_eval["confusion_matrix"]
        }
    }

    report_path = Path(output_dir) / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation report saved to {report_path}")
    print(f"\nEvaluation metrics successfully written to {report_path}")


if __name__ == "__main__":
    main()
