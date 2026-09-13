"""Train and evaluate baseline and transformer symptom identification models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import numpy as np
from zenova.symptoms.taxonomy import SYMPTOM_SIGNALS
from zenova.symptoms.baseline import SymptomTfidfBaseline
from zenova.symptoms.trainer import SymptomTransformerTrainer
from zenova.symptoms.evaluator import SymptomModelEvaluator
from zenova.symptoms.inference import SymptomInferenceEngine
from zenova.core.logging import get_logger

logger = get_logger("zenova.scripts.train_symptoms")


def load_jsonl(filepath: str):
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def main():
    processed_dir = Path("data/processed/psysym_mental_health_symptoms")
    train_file = processed_dir / "train.jsonl"
    val_file = processed_dir / "val.jsonl"
    test_file = processed_dir / "test.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(f"Missing train split at {train_file}. Run scripts/ingest_psysym.py first.")

    train_data = load_jsonl(str(train_file))
    val_data = load_jsonl(str(val_file))
    test_data = load_jsonl(str(test_file))

    train_texts = [s["text"] for s in train_data]
    train_labels = [s["labels"] for s in train_data]

    val_texts = [s["text"] for s in val_data]
    val_labels = [s["labels"] for s in val_data]

    test_texts = [s["text"] for s in test_data]
    test_labels = [s["labels"] for s in test_data]

    labels = sorted(SYMPTOM_SIGNALS)
    logger.info(f"Loaded splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)} across {len(labels)} symptom classes.")

    output_dir = "models/symptoms"

    # Build binary ground truth matrix for test split
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    y_test_true = np.zeros((len(test_data), len(labels)), dtype=int)
    for i, s_labels in enumerate(test_labels):
        for lbl in s_labels:
            if lbl in label2id:
                y_test_true[i, label2id[lbl]] = 1

    # =========================================================================
    # 1. Train and Evaluate Multi-Label TF-IDF Baseline
    # =========================================================================
    print("\n--- 1. Training Multi-Label TF-IDF Baseline Classifier ---")
    baseline = SymptomTfidfBaseline(labels=labels)
    baseline.train(train_texts, train_labels)
    baseline.save(output_dir)

    baseline_preds = baseline.predict(test_texts, threshold=0.50)
    baseline_eval = SymptomModelEvaluator.evaluate(y_test_true, baseline_preds, labels=labels)

    print(f"Baseline Micro F1: {baseline_eval['micro_f1']:.4f}")
    print(f"Baseline Macro F1: {baseline_eval['macro_f1']:.4f}")
    print(f"Baseline Subset Accuracy: {baseline_eval['subset_accuracy']:.4f}")
    print(f"Baseline Hamming Loss: {baseline_eval['hamming_loss']:.4f}")
    print("Baseline Per-Label F1:")
    for lbl, m in baseline_eval["per_label"].items():
        print(f"  • {lbl:35s}: F1={m['f1']:.4f} (P={m['precision']:.4f}, R={m['recall']:.4f}, n={m['support']})")

    # =========================================================================
    # 2. Train and Evaluate PyTorch Multi-Label Transformer
    # =========================================================================
    print("\n--- 2. Training PyTorch Multi-Label Transformer Sequence Classifier ---")
    trainer = SymptomTransformerTrainer(labels=labels, max_vocab=6000, max_len=128)
    transformer_model = trainer.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        epochs=6,
        batch_size=32,
        lr=1e-3,
        device="cpu"
    )
    trainer.save_artifacts(transformer_model, output_dir)

    # Evaluate Transformer via Inference Engine on Test Split
    inference_engine = SymptomInferenceEngine(model_dir=output_dir)

    transformer_preds = np.zeros((len(test_texts), len(labels)), dtype=int)
    for i, t in enumerate(test_texts):
        res = inference_engine.predict(t, threshold=0.50)
        for sig in res["signals"]:
            if sig["marker_name"] in label2id:
                transformer_preds[i, label2id[sig["marker_name"]]] = 1

    transformer_eval = SymptomModelEvaluator.evaluate(y_test_true, transformer_preds, labels=labels)

    print(f"\nTransformer Micro F1: {transformer_eval['micro_f1']:.4f}")
    print(f"Transformer Macro F1: {transformer_eval['macro_f1']:.4f}")
    print(f"Transformer Subset Accuracy: {transformer_eval['subset_accuracy']:.4f}")
    print(f"Transformer Hamming Loss: {transformer_eval['hamming_loss']:.4f}")
    print("Transformer Per-Label F1:")
    for lbl, m in transformer_eval["per_label"].items():
        print(f"  • {lbl:35s}: F1={m['f1']:.4f} (P={m['precision']:.4f}, R={m['recall']:.4f}, n={m['support']})")

    # =========================================================================
    # 3. Save Comparative Evaluation Report
    # =========================================================================
    report = {
        "dataset": "PsySym Mental Health Symptoms",
        "splits": {
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data)
        },
        "labels": labels,
        "models": {
            "baseline_tfidf": baseline_eval,
            "transformer": transformer_eval
        }
    }

    report_path = Path(output_dir) / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved comparative evaluation report to {report_path}")

    # Canonical benchmark demonstration
    print("\n--- Canonical Benchmark Verification ---")
    prompt_test = "I haven't been sleeping properly and I don't enjoy things anymore."
    result = inference_engine.predict(prompt_test)
    print(f"Input: \"{prompt_test}\"")
    print(f"Detected Signals: {[s['marker_name'] for s in result['signals']]}")
    print(f"Output Payload:\n{json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
