import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import random
import urllib.request
from collections import defaultdict
from typing import List, Dict, Any
from zenova.data.pipeline import GenericDatasetPipeline
from zenova.core.logging import get_logger

logger = get_logger("zenova.scripts.ingest_goemotions")

BASE_URL = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data"


def fetch_file(filename: str) -> str:
    url = f"{BASE_URL}/{filename}"
    logger.info(f"Fetching {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def parse_goemotions() -> List[Dict[str, Any]]:
    emotions_txt = fetch_file("emotions.txt").strip().splitlines()
    ekman_json_str = fetch_file("ekman_mapping.json")
    ekman_map = json.loads(ekman_json_str)

    fine_to_ekman = {}
    for ekman_cat, fine_list in ekman_map.items():
        for fine in fine_list:
            fine_to_ekman[fine] = ekman_cat
    fine_to_ekman["neutral"] = "neutral"

    all_samples = []
    seen_texts = set()

    for split_name in ["train.tsv", "dev.tsv", "test.tsv"]:
        raw_tsv = fetch_file(split_name)
        lines = raw_tsv.strip().splitlines()
        logger.info(f"Processing {split_name} ({len(lines)} lines)...")

        for line in lines:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            text = parts[0].strip()
            t_lower = text.lower()
            if t_lower in seen_texts:
                continue

            raw_indices = parts[1].strip().split(",")
            fine_names = [emotions_txt[int(idx)] for idx in raw_indices if idx.isdigit() and int(idx) < len(emotions_txt)]
            ekman_labels = list(set(fine_to_ekman.get(f) for f in fine_names if f in fine_to_ekman))

            if len(ekman_labels) == 1 and ekman_labels[0]:
                seen_texts.add(t_lower)
                all_samples.append({
                    "text": text,
                    "label": ekman_labels[0],
                    "fine_emotions": fine_names,
                    "source_split": split_name.replace(".tsv", "")
                })

    logger.info(f"Parsed {len(all_samples)} unique, unambiguous Ekman-mapped emotion samples.")
    return all_samples


def main():
    dataset_name = "GoEmotions Ekman"
    samples = parse_goemotions()

    random.seed(42)
    by_label = defaultdict(list)
    for s in samples:
        by_label[s["label"]].append(s)

    subset = []
    for lbl, s_list in by_label.items():
        random.shuffle(s_list)
        subset.extend(s_list[:1500])

    random.shuffle(subset)
    samples = subset
    logger.info(f"Selected balanced stratified sample of {len(samples)} unique utterances for training.")

    pipeline = GenericDatasetPipeline(dataset_name=dataset_name)

    metadata = pipeline.run(
        raw_samples=samples,
        source="Google Research (Demszky et al., ACL 2020)",
        official_url="https://github.com/google-research/google-research/tree/master/goemotions",
        license_name="Apache-2.0",
        citation="Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. In ACL 2020, pp. 4040-4054.",
        intended_task="emotion_analysis",
        features=["text", "fine_emotions"],
        labels=["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"],
        label_field="label",
        text_field="text",
        version="1.0.0",
        seed=42,
        explicit_non_goals=[
            "Psychiatric disorder diagnosis",
            "Depression / Bipolar diagnostic classification",
            "Suicide risk assessment"
        ]
    )

    print("\n=== GoEmotions Ingestion Complete ===")
    print(f"Dataset: {metadata.dataset_name} (v{metadata.version})")
    print(f"Total Samples: {metadata.number_of_samples['processed']}")
    print(f"Train Count: {metadata.train_validation_test_split.train_count}")
    print(f"Val Count: {metadata.train_validation_test_split.val_count}")
    print(f"Test Count: {metadata.train_validation_test_split.test_count}")
    print(f"Leakage Check: {'PASSED' if metadata.leakage_check_passed else 'FAILED'}")
    print(f"Datasheet: {metadata.local_storage_location['docs']}")


if __name__ == "__main__":
    main()
