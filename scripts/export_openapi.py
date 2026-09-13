"""Export static OpenAPI schema from FastAPI app to docs/openapi.json."""
import os
import sys
import json
from pathlib import Path

# Add src to pythonpath
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from zenova.api.app import app
from zenova.core.logging import get_logger

logger = get_logger("zenova.scripts.openapi")


def export_openapi(output_path: str = "docs/openapi.json") -> Path:
    """Generate and write static OpenAPI schema."""
    target_file = root_dir / output_path
    target_file.parent.mkdir(parents=True, exist_ok=True)

    schema = app.openapi()
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    logger.info(f"OpenAPI schema successfully exported to {target_file} ({len(schema.get('paths', {}))} endpoints documented)")
    return target_file


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/openapi.json"
    export_openapi(out)
