"""Train and evaluate baseline and transformer crisis risk models with False-Negative analysis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
from zenova.risk.taxonomy import RISK_LEVELS
from zenova.risk.baseline import RiskTfidfBaseline
from zenova.risk.trainer import RiskTransformerTrainer
from zenova.risk.evaluator import RiskModelEvaluator
from zenova.risk.inference import RiskInferenceEngine
from zenova.core.logging import get_logger

logger = get_logger("zenova.scripts.train_risk")


def load_jsonl(filepath: str):
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def main():
    processed_dir = Path("data/processed/c_ssrs_suicide_and_crisis_risk_benchmark")
    train_file = processed_dir / "train.jsonl"
    val_file = processed_dir / "val.jsonl"
    test_file = processed_dir / "test.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(f"Missing train split at {train_file}. Run scripts/ingest_crisis.py first.")

    train_data = load_jsonl(str(train_file))
    val_data = load_jsonl(str(val_file))
    test_data = load_jsonl(str(test_file))

    train_texts = [s["text"] for s in train_data]
    train_labels = [s["risk_level"] for s in train_data]

    val_texts = [s["text"] for s in val_data]
    val_labels = [s["risk_level"] for s in val_data]

    test_texts = [s["text"] for s in test_data]
    test_labels = [s["risk_level"] for s in test_data]

    labels = RISK_LEVELS
    logger.info(f"Loaded splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)} across {labels}")

    output_dir = "models/risk"

    # =========================================================================
    # 1. Train and Evaluate Cost-Sensitive TF-IDF Baseline
    # =========================================================================
    print("\n--- 1. Training Cost-Sensitive TF-IDF Baseline Classifier ---")
    baseline = RiskTfidfBaseline(labels=labels)
    baseline.train(train_texts, train_labels)
    baseline.save(output_dir)

    baseline_preds = baseline.predict(test_texts, crisis_threshold=0.35)
    baseline_eval = RiskModelEvaluator.evaluate(test_labels, baseline_preds, texts=test_texts, labels=labels)

    print(f"Baseline Accuracy: {baseline_eval['accuracy']:.4f}")
    print(f"Baseline Macro F1: {baseline_eval['macro_f1']:.4f}")
    print(f"Baseline Specificity (Low Risk): {baseline_eval['specificity_low_risk']:.4f}")
    print(f"Baseline High/Critical Sensitivity: {baseline_eval['safety_sensitivity_high_critical']:.4f}")
    print(f"Baseline False Negatives on High/Critical: {baseline_eval['total_false_negatives_count']}")
    print("Baseline Per-Class Metrics:")
    for lbl, m in baseline_eval["per_class"].items():
        print(f"  • {lbl:12s}: F1={m['f1']:.4f} (P={m['precision']:.4f}, R={m['recall']:.4f}, n={m['support']})")

    # =========================================================================
    # 2. Train and Evaluate PyTorch Cost-Sensitive Risk Transformer
    # =========================================================================
    print("\n--- 2. Training PyTorch Cost-Sensitive Risk Transformer Sequence Classifier ---")
    trainer = RiskTransformerTrainer(labels=labels, max_vocab=5000, max_len=128)
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
    inference_engine = RiskInferenceEngine(model_dir=output_dir)
    transformer_preds = [inference_engine.predict(t)["risk_level"] for t in test_texts]
    transformer_eval = RiskModelEvaluator.evaluate(test_labels, transformer_preds, texts=test_texts, labels=labels)

    print(f"\nTransformer Accuracy: {transformer_eval['accuracy']:.4f}")
    print(f"Transformer Macro F1: {transformer_eval['macro_f1']:.4f}")
    print(f"Transformer Specificity (Low Risk): {transformer_eval['specificity_low_risk']:.4f}")
    print(f"Transformer High/Critical Sensitivity: {transformer_eval['safety_sensitivity_high_critical']:.4f}")
    print(f"Transformer False Negatives on High/Critical: {transformer_eval['total_false_negatives_count']}")
    print("Transformer Per-Class Metrics:")
    for lbl, m in transformer_eval["per_class"].items():
        print(f"  • {lbl:12s}: F1={m['f1']:.4f} (P={m['precision']:.4f}, R={m['recall']:.4f}, n={m['support']})")

    # Confusion Matrix
    print("\nTransformer Confusion Matrix (Rows=True, Cols=Pred):")
    print(f"Labels: {labels}")
    for row in transformer_eval["confusion_matrix"]["matrix"]:
        print(f"  {row}")

    # =========================================================================
    # 3. Save Comparative Evaluation Report
    # =========================================================================
    report = {
        "dataset": "C-SSRS Suicide and Crisis Risk Benchmark",
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

    # =========================================================================
    # 4. Adversarial and Edge-Case Stress Testing
    # =========================================================================
    print("\n--- Adversarial and Edge-Case Stress Testing ---")
    stress_cases = [
        ("This traffic is killing me right now!", "low"),
        ("I'm dying of laughter, that was the funniest joke ever!", "low"),
        ("I feel completely overwhelmed by work and need a break.", "low"),
        ("I don't want to wake up anymore, everything is so heavy.", "moderate"),
        ("I wish I could go to sleep and never wake up again.", "moderate"),
        ("I want to end my life, I'm planning my suicide tonight.", "high"),
        ("I can't take this pain anymore, I'm going to kill myself.", "high"),
        ("I have the pills right next to me and I'm swallowing them now. Goodbye.", "critical"),
        ("Standing on the ledge of the bridge right now and jumping.", "critical")
    ]

    for text, expected in stress_cases:
        res = inference_engine.predict(text)
        pred = res["risk_level"]
        esc = res["requires_escalation"]
        status = "PASSED" if pred == expected else "FAILED"
        print(f"[{status}] Input: \"{text}\"")
        print(f"         Expected: {expected} | Predicted: {pred} | Escalation: {esc} | Conf: {res['confidence']}")


if __name__ == "__main__":
    main()
