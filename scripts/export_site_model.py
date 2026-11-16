"""Export the fixed experiment into the local frontend without retraining."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=ROOT / "frontend" / "dist")
    args = parser.parse_args()
    if not (args.dist / "index.html").is_file():
        raise SystemExit("The destination must be an existing frontend dist directory.")
    model = json.loads((ROOT / "artifacts" / "browser-model.json").read_text(encoding="utf-8"))
    results = json.loads((ROOT / "reports" / "model_results.json").read_text(encoding="utf-8"))
    source = "window.BIKE_MODEL=" + json.dumps(model, separators=(",", ":"), allow_nan=False) + ";\nwindow.MODEL_RESULTS=" + json.dumps(results, separators=(",", ":"), allow_nan=False) + ";"
    (args.dist / "model.js").write_text(source, encoding="utf-8")
    print("Exported", results["selected"], "to", args.dist / "model.js")


if __name__ == "__main__":
    main()
