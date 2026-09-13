import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
"""Ingestion script for ESConv dataset."""
import os
import json
import shutil
from collections import Counter
from zenova.strategy.dataset import ESConvDatasetLoader
from zenova.core.logging import get_logger

logger = get_logger("zenova.scripts.ingest_esconv")


def main():
    raw_src = "ESConv.json"
    raw_dest = "data/raw/esconv/ESConv.json"
    output_dir = "data/processed/strategy"

    os.makedirs(os.path.dirname(raw_dest), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(raw_dest) and os.path.exists(raw_src):
        shutil.copyfile(raw_src, raw_dest)
        logger.info(f"Copied {raw_src} -> {raw_dest}")

    loader = ESConvDatasetLoader(raw_data_path=raw_dest)
    dialogues = loader.load_raw()
    samples = loader.extract_strategy_samples(dialogues)

    train_samples, val_samples, test_samples = loader.create_splits(samples, seed=42)

    def write_jsonl(filename: str, sample_list):
        out_p = os.path.join(output_dir, filename)
        with open(out_p, "w", encoding="utf-8") as f:
            for s in sample_list:
                f.write(json.dumps(s.to_dict()) + "\n")
        logger.info(f"Saved {len(sample_list)} samples to {out_p}")

    write_jsonl("train.jsonl", train_samples)
    write_jsonl("val.jsonl", val_samples)
    write_jsonl("test.jsonl", test_samples)

    # Strategy distribution statistics
    all_counts = Counter(s.target_strategy.value for s in samples)
    train_counts = Counter(s.target_strategy.value for s in train_samples)
    val_counts = Counter(s.target_strategy.value for s in val_samples)
    test_counts = Counter(s.target_strategy.value for s in test_samples)

    summary = {
        "total_dialogues": len(dialogues),
        "total_supporter_strategy_turns": len(samples),
        "splits": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples)
        },
        "strategy_distribution_total": dict(all_counts),
        "strategy_distribution_train": dict(train_counts),
        "strategy_distribution_val": dict(val_counts),
        "strategy_distribution_test": dict(test_counts),
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Ingestion summary written to {summary_path}")
    print("\n=== ESConv Ingestion Summary ===")
    print(f"Total Dialogues: {len(dialogues)}")
    print(f"Total Annotated Supporter Turns: {len(samples)}")
    print(f"Train Turns: {len(train_samples)}")
    print(f"Val Turns: {len(val_samples)}")
    print(f"Test Turns: {len(test_samples)}")
    print("\nStrategy Frequencies:")
    for strat, count in all_counts.most_common():
        pct = (count / len(samples)) * 100
        print(f"  • {strat:30s}: {count:5d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
