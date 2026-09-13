"""Controlled Ablation Experiments and Training Script for Support Strategy Models.

Ablation Conditions:
  Condition A: Conversation only (dialogue history + current message)
  Condition B: Conversation + Emotion affect signal
  Condition C: Conversation + Emotion + Additional context (situation description, problem type)

Models:
  1. TF-IDF + Logistic Regression Baseline
  2. PyTorch Self-Attention Transformer Classifier
"""
import os
import sys
import json
from pathlib import Path
from typing import List

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zenova.strategy.dataset import ESConvTurnSample
from zenova.strategy.taxonomy import StrategyTaxonomy
from zenova.strategy.baseline import StrategyTfidfBaseline
from zenova.strategy.trainer import StrategyTransformerTrainer
from zenova.strategy.evaluator import StrategyEvaluator
from zenova.schemas.strategy import SupportStrategy, DialogStage
from zenova.core.logging import get_logger

logger = get_logger("zenova.scripts.train_strategy")


def load_jsonl_samples(filepath: str) -> List[ESConvTurnSample]:
    """Load ESConvTurnSample objects from JSONL split."""
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            sample = ESConvTurnSample(
                dialog_id=d["dialog_id"],
                turn_index=d["turn_index"],
                situation=d.get("situation", ""),
                emotion_type=d.get("emotion_type", ""),
                problem_type=d.get("problem_type", ""),
                dialog_history=d.get("dialog_history", []),
                target_strategy=SupportStrategy(d["target_strategy"]),
                supporter_response=d.get("supporter_response", ""),
                stage=DialogStage(d.get("stage", DialogStage.COMFORTING.value)),
                current_user_message=d.get("current_user_message", "")
            )
            samples.append(sample)
    return samples


def main():
    processed_dir = Path("data/processed/strategy")
    model_dir = Path("models/strategy")
    exp_dir = Path("experiments")
    model_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "train.jsonl"
    val_path = processed_dir / "val.jsonl"
    test_path = processed_dir / "test.jsonl"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        raise FileNotFoundError(f"Processed datasets not found in {processed_dir}. Run scripts/ingest_esconv.py first.")

    logger.info("Loading conversation-split dataset turns...")
    train_samples = load_jsonl_samples(str(train_path))
    val_samples = load_jsonl_samples(str(val_path))
    test_samples = load_jsonl_samples(str(test_path))

    taxonomy = StrategyTaxonomy.default()
    evaluator = StrategyEvaluator(taxonomy=taxonomy)

    logger.info(
        f"Loaded splits: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}"
    )

    ablation_results = {
        "dataset": "ESConv",
        "splits": {
            "train_turns": len(train_samples),
            "val_turns": len(val_samples),
            "test_turns": len(test_samples)
        },
        "experiments": {}
    }

    conditions = ["A", "B", "C"]
    condition_names = {
        "A": "Conversation Only",
        "B": "Conversation + Emotion",
        "C": "Conversation + Emotion + Context (Full Multimodal)"
    }

    # -------------------------------------------------------------
    # 1. BASELINE EXPERIMENTS: TF-IDF + Logistic Regression
    # -------------------------------------------------------------
    logger.info("\n" + "=" * 60 + "\nRUNNING BASELINE (TF-IDF + LOGISTIC REGRESSION) EXPERIMENTS\n" + "=" * 60)
    best_baseline = None
    best_baseline_f1 = -1.0

    for cond in conditions:
        logger.info(f"\n--- Baseline Condition {cond}: {condition_names[cond]} ---")
        train_texts = [s.format_input(condition=cond) for s in train_samples]
        train_labels = [s.target_strategy.value for s in train_samples]

        val_texts = [s.format_input(condition=cond) for s in val_samples]
        val_labels = [s.target_strategy.value for s in val_samples]

        test_texts = [s.format_input(condition=cond) for s in test_samples]
        test_labels = [s.target_strategy.value for s in test_samples]

        baseline = StrategyTfidfBaseline(taxonomy=taxonomy)
        baseline.train(train_texts, train_labels)

        # Val eval
        val_preds = baseline.predict(val_texts)
        val_probs = baseline.predict_proba(val_texts)
        val_metrics = evaluator.evaluate(val_labels, val_preds, val_probs)

        # Test eval
        test_preds = baseline.predict(test_texts)
        test_probs = baseline.predict_proba(test_texts)
        test_metrics = evaluator.evaluate(test_labels, test_preds, test_probs)

        exp_key = f"baseline_condition_{cond.lower()}"
        ablation_results["experiments"][exp_key] = {
            "model": "TF-IDF + LogisticRegression",
            "condition": cond,
            "condition_description": condition_names[cond],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_precision": test_metrics["macro_precision"],
            "test_macro_recall": test_metrics["macro_recall"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_ece": test_metrics.get("ece", 0.0),
            "test_per_class": test_metrics["per_class"],
            "test_confusion_matrix": test_metrics["confusion_matrix"]
        }

        logger.info(
            f"Baseline Condition {cond} -> Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['macro_f1']:.4f} | "
            f"Test Acc: {test_metrics['accuracy']:.4f}, Test F1: {test_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_baseline_f1:
            best_baseline_f1 = val_metrics["macro_f1"]
            best_baseline = baseline

    # Save best baseline model
    if best_baseline is not None:
        best_baseline.save(str(model_dir))
        logger.info(f"Saved best baseline model to {model_dir / 'tfidf_baseline.joblib'}")

    # -------------------------------------------------------------
    # 2. TRANSFORMER EXPERIMENTS: PyTorch Self-Attention Model
    # -------------------------------------------------------------
    logger.info("\n" + "=" * 60 + "\nRUNNING TRANSFORMER (SELF-ATTENTION) EXPERIMENTS\n" + "=" * 60)
    best_transformer = None
    best_transformer_f1 = -1.0
    best_trans_test_metrics = None

    for cond in conditions:
        logger.info(f"\n--- Transformer Condition {cond}: {condition_names[cond]} ---")
        trainer = StrategyTransformerTrainer(
            taxonomy=taxonomy,
            max_vocab=6000,
            max_len=64,
            d_model=64,
            nhead=2,
            num_layers=1,
            dim_feedforward=128,
            dropout=0.1
        )
        val_metrics = trainer.train(
            train_samples=train_samples,
            val_samples=val_samples,
            condition=cond,
            epochs=2,
            batch_size=128,
            lr=1e-3,
            device="cpu"
        )

        test_metrics = trainer.evaluate_samples(
            samples=test_samples,
            condition=cond,
            batch_size=128,
            device="cpu"
        )

        exp_key = f"transformer_condition_{cond.lower()}"
        ablation_results["experiments"][exp_key] = {
            "model": "Transformer (PyTorch Self-Attention)",
            "condition": cond,
            "condition_description": condition_names[cond],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_precision": test_metrics["macro_precision"],
            "test_macro_recall": test_metrics["macro_recall"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_ece": test_metrics.get("ece", 0.0),
            "test_per_class": test_metrics["per_class"],
            "test_confusion_matrix": test_metrics["confusion_matrix"]
        }

        logger.info(
            f"Transformer Condition {cond} -> Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['macro_f1']:.4f} | "
            f"Test Acc: {test_metrics['accuracy']:.4f}, Test F1: {test_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_transformer_f1:
            best_transformer_f1 = val_metrics["macro_f1"]
            best_transformer = trainer
            best_trans_test_metrics = test_metrics

    # Save best transformer model
    if best_transformer is not None:
        best_transformer.save(str(model_dir))
        logger.info(f"Saved best transformer checkpoint and artifacts to {model_dir}")

    # Write ablation comparison
    ablation_path = exp_dir / "strategy_ablation_results.json"
    with open(ablation_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
    logger.info(f"Saved ablation results to {ablation_path}")

    # Save test evaluation report
    test_report_path = model_dir / "test_evaluation_report.json"
    with open(test_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_transformer_test": best_trans_test_metrics,
            "taxonomy": taxonomy.to_dict()
        }, f, indent=2)
    logger.info(f"Saved test evaluation report to {test_report_path}")

    # Print summary comparison table
    print("\n" + "=" * 80)
    print("                ESConv STRATEGY MODEL ABLATION SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<32} | {'Val Acc':<8} | {'Val F1':<8} | {'Test Acc':<8} | {'Test F1':<8}")
    print("-" * 80)
    for exp_k, res in ablation_results["experiments"].items():
        print(
            f"{exp_k:<32} | {res['val_accuracy']:<8.4f} | {res['val_macro_f1']:<8.4f} | "
            f"{res['test_accuracy']:<8.4f} | {res['test_macro_f1']:<8.4f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
