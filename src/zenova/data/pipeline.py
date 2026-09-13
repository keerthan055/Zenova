"""Generic end-to-end dataset pipeline orchestrator."""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from zenova.core.logging import get_logger
from zenova.data.storage import DatasetStorage
from zenova.data.metadata import DatasetMetadata, DatasetSplitInfo
from zenova.data.integrity import DatasetIntegrityChecker
from zenova.data.preprocessor import BaseDatasetPreprocessor, TextDialoguePreprocessor
from zenova.data.splitter import DatasetSplitter
from zenova.data.leakage import DatasetLeakageDetector
from zenova.data.statistics import DatasetStatisticsGenerator
from zenova.data.docs_generator import DatasetDocumentationGenerator
from zenova.data.registry import DatasetRegistry
from zenova.data.provenance import ProvenanceAuditor

logger = get_logger("zenova.data.pipeline")


class GenericDatasetPipeline:
    """Standard, reusable pipeline for acquiring, validating, splitting, profiling, and registering datasets."""

    def __init__(self, dataset_name: str, base_data_dir: str = "data"):
        self.dataset_name = dataset_name
        self.storage = DatasetStorage(base_data_dir=base_data_dir)
        self.paths = self.storage.get_dataset_paths(dataset_name)
        self.registry = DatasetRegistry(metadata_dir=str(self.storage.metadata_dir))

    def run(
        self,
        raw_samples: List[Dict[str, Any]],
        source: str,
        official_url: str,
        license_name: str,
        citation: str,
        intended_task: str,
        features: List[str],
        labels: List[str],
        label_field: str = "label",
        text_field: str = "text",
        group_field: Optional[str] = None,
        preprocessor: Optional[BaseDatasetPreprocessor] = None,
        explicit_non_goals: Optional[List[str]] = None,
        version: str = "1.0.0",
        seed: int = 42
    ) -> DatasetMetadata:
        logger.info(f"Starting GenericDatasetPipeline for '{self.dataset_name}' (v{version})...")

        # 1. Audit provenance and license
        lic_ok, lic_msg = ProvenanceAuditor.audit_license(license_name)
        if not lic_ok:
            raise ValueError(f"License audit failed: {lic_msg}")
        logger.info(lic_msg)

        # 2. Save raw data
        raw_file = self.paths["raw"] / "raw_data.jsonl"
        with open(raw_file, "w", encoding="utf-8") as f:
            for s in raw_samples:
                f.write(json.dumps(s) + "\n")
        raw_checksum = DatasetIntegrityChecker.compute_sha256(str(raw_file))

        # 3. Preprocessing
        prep = preprocessor or TextDialoguePreprocessor(text_field=text_field, target_field=label_field)
        processed_samples, prep_stats = prep.preprocess_batch(raw_samples)
        logger.info(f"Preprocessed: {prep_stats['processed']} retained, {prep_stats['dropped']} dropped.")

        # Save interim
        interim_file = self.paths["interim"] / "cleaned_data.jsonl"
        with open(interim_file, "w", encoding="utf-8") as f:
            for s in processed_samples:
                f.write(json.dumps(s) + "\n")
        interim_checksum = DatasetIntegrityChecker.compute_sha256(str(interim_file))

        # 4. Reproducible Splitting
        if group_field:
            train_s, val_s, test_s = DatasetSplitter.split_by_group(
                processed_samples, group_field=group_field, seed=seed
            )
            split_strategy = f"grouped_by_{group_field}"
        else:
            train_s, val_s, test_s = DatasetSplitter.split_stratified(
                processed_samples, label_field=label_field, seed=seed
            )
            split_strategy = "stratified"

        # 5. Leakage Detection Audit
        leakage_passed = True
        leakage_reports = {}
        if group_field:
            grp_passed, grp_rep = DatasetLeakageDetector.check_group_leakage(
                train_s, val_s, test_s, group_field=group_field
            )
            leakage_passed = leakage_passed and grp_passed
            leakage_reports["group_leakage"] = grp_rep

        txt_passed, txt_rep = DatasetLeakageDetector.check_exact_text_leakage(
            train_s, test_s, text_field=text_field
        )
        leakage_passed = leakage_passed and txt_passed
        leakage_reports["text_leakage"] = txt_rep

        if not leakage_passed:
            logger.warning(f"Data leakage detected in '{self.dataset_name}': {leakage_reports}")
        else:
            logger.info("Data leakage audit PASSED with zero cross-split overlap.")

        # 6. Save Processed Splits
        split_files = {
            "train": self.paths["processed"] / "train.jsonl",
            "val": self.paths["processed"] / "val.jsonl",
            "test": self.paths["processed"] / "test.jsonl"
        }
        checksums = {
            "raw": raw_checksum,
            "interim": interim_checksum
        }
        for split_name, s_list in [("train", train_s), ("val", val_s), ("test", test_s)]:
            out_file = split_files[split_name]
            with open(out_file, "w", encoding="utf-8") as f:
                for s in s_list:
                    f.write(json.dumps(s) + "\n")
            checksums[split_name] = DatasetIntegrityChecker.compute_sha256(str(out_file))

        # 7. Compute Statistics
        stats = DatasetStatisticsGenerator.compute_statistics(
            processed_samples, label_field=label_field, text_field=text_field
        )
        stats["leakage_audit"] = leakage_reports

        # 8. Compile Metadata
        split_info = DatasetSplitInfo(
            train_count=len(train_s),
            val_count=len(val_s),
            test_count=len(test_s),
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            random_seed=seed,
            split_strategy=split_strategy
        )

        metadata = DatasetMetadata(
            dataset_name=self.dataset_name,
            version=version,
            source=source,
            official_url=official_url,
            license=license_name,
            citation=citation,
            download_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            local_storage_location={k: str(v) for k, v in self.paths.items()},
            intended_task=intended_task,
            features=features,
            labels=labels,
            number_of_samples={
                "total": len(raw_samples),
                "raw": len(raw_samples),
                "processed": len(processed_samples)
            },
            train_validation_test_split=split_info,
            preprocessing_steps=[
                "Text normalization (clean_text)",
                "Minimum character filtering",
                f"Split ({split_strategy}, seed={seed})"
            ],
            checksums=checksums,
            leakage_check_passed=leakage_passed,
            statistics=stats,
            explicit_non_goals=explicit_non_goals or [
                "Clinical diagnostic classification",
                "Automated medication prescribing"
            ]
        )

        # 9. Generate Datasheet Documentation
        docs_file = self.paths["docs"]
        DatasetDocumentationGenerator.generate_markdown_datasheet(metadata, str(docs_file))
        logger.info(f"Generated datasheet at {docs_file}")

        # 10. Register in Catalog
        self.registry.register_dataset(metadata)
        logger.info(f"GenericDatasetPipeline successfully completed for '{self.dataset_name}'.")

        return metadata
