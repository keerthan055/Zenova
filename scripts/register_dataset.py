import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
"""CLI script to register and validate new research datasets for ZENOVA."""
import argparse
import json
import sys
from pathlib import Path
from zenova.data.registry import DatasetRegistry, DatasetMetadata
from zenova.core.logging import get_logger

logger = get_logger("zenova.scripts.register_dataset")


def main():
    parser = argparse.ArgumentParser(description="Register a research dataset into ZENOVA registry")
    parser.add_argument("--name", required=True, help="Canonical dataset name")
    parser.add_argument("--version", default="1.0.0", help="Dataset version")
    parser.add_argument("--source", required=True, help="Authoritative source or institution")
    parser.add_argument("--citation", required=True, help="Full academic citation")
    parser.add_argument("--license", required=True, help="Permitted license (e.g., Apache-2.0, MIT, CC-BY-4.0)")
    parser.add_argument("--task", required=True, help="Exact ML task (e.g., support_strategy_planning, emotion_analysis)")
    parser.add_argument("--file-path", required=True, help="Local path to raw or processed dataset")
    parser.add_argument("--repo-url", default="", help="Public repository or publication URL")

    args = parser.parse_args()

    # Rule checks
    if not args.license or args.license.lower() in ["unknown", "unspecified", "none"]:
        logger.error("Dataset Rule Violation: Provenance and license must be explicitly identified.")
        sys.exit(1)

    metadata = DatasetMetadata(
        name=args.name,
        version=args.version,
        source=args.source,
        citation=args.citation,
        license=args.license,
        repository_url=args.repo_url if args.repo_url else None,
        file_path=args.file_path,
        intended_task=args.task,
        explicit_non_goals=[
            "Clinical diagnosis replacement",
            "Suicide risk prediction unless validated explicitly for this task"
        ]
    )

    registry = DatasetRegistry()
    registry.register_dataset(metadata)
    print(f"\nSuccessfully registered dataset '{metadata.name}' for task '{metadata.intended_task}'.")


if __name__ == "__main__":
    main()
