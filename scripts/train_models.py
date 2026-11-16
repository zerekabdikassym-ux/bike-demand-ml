"""Train and select on validation; evaluate the locked winner once on test.

No temperature conversion is assumed. Only calendar and weather-category inputs
are used. Run from anywhere with the project's isolated Python environment.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

# Keep training bounded on a laptop.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["hr", "weekday", "mnth", "weathersit", "holiday", "workingday", "yr"]


def metrics(y, pred):
    pred = np.maximum(0, pred)
    return {"mae": float(mean_absolute_error(y, pred)), "rmse": float(root_mean_squared_error(y, pred))}


def export_model(model):
    if isinstance(model, GradientBoostingRegressor):
        trees = []
        for estimator in model.estimators_.ravel():
            tree = estimator.tree_
            trees.append([[int(tree.children_left[i]), int(tree.children_right[i]), int(tree.feature[i]), float(tree.threshold[i]), float(tree.value[i, 0, 0])] for i in range(tree.node_count)])
        return {"kind": "gradient_boosting", "base": float(model.init_.constant_[0, 0]), "rate": model.learning_rate, "trees": trees}
    if isinstance(model, HistGradientBoostingRegressor):
        # Version-locked sklearn internals: checked against Python predictions
        # over every train/validation/test row before the export is accepted.
        trees = []
        for iteration in model._predictors:
            nodes = iteration[0].nodes
            trees.append([[int(n["left"]), int(n["right"]), int(n["feature_idx"]), float(n["num_threshold"]), float(n["value"]), int(n["is_leaf"])] for n in nodes])
        return {"kind": "hist_gradient_boosting", "base": float(model._baseline_prediction[0, 0]), "rate": 1.0, "trees": trees}
    encoder, ridge = model.steps[0][1], model.steps[1][1]
    return {"kind": "ridge", "categories": [v.tolist() for v in encoder.categories_], "coefficients": ridge.coef_.tolist(), "intercept": float(ridge.intercept_)}


def predict_export(export, x):
    if export["kind"] == "ridge":
        result = export["intercept"]
        offset = 0
        for value, categories in zip(x, export["categories"]):
            if value in categories:
                result += export["coefficients"][offset + categories.index(value)]
            offset += len(categories)
        return max(0.0, result)
    result = export["base"]
    for tree in export["trees"]:
        node = tree[0]
        while not (node[5] if export["kind"] == "hist_gradient_boosting" else node[0] == -1):
            # All features are finite integers, so no missing-value routing.
            node = tree[node[0] if x[node[2]] <= node[3] else node[1]]
        result += export["rate"] * node[4]
    return max(0.0, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace-results", action="store_true", help="Explicitly replace a previous fixed experiment; do not tune using its test results")
    args = parser.parse_args()
    results_path = ROOT / "reports" / "model_results.json"
    if results_path.exists() and not args.replace_results:
        raise SystemExit("Experiment already evaluated. Existing results preserved; do not repeatedly tune on the test.")
    path = ROOT / "data" / "raw" / "hour.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    split = json.loads((ROOT / "reports" / "split_manifest.json").read_text())
    parts = {name: [r for r in rows if bounds["start"] <= r["dteday"] <= bounds["end"]] for name, bounds in split.items()}
    for name, values in parts.items():
        assert len(values) == split[name]["rows"]
    x = {name: np.array([[int(r[f]) for f in FEATURES] for r in values], dtype=np.float64) for name, values in parts.items()}
    y = {name: np.array([int(r["cnt"]) for r in values]) for name, values in parts.items()}
    baseline = {}
    for r in parts["train"]:
        key = (int(r["weekday"]), int(r["hr"]))
        baseline.setdefault(key, []).append(int(r["cnt"]))
    baseline = {key: float(np.mean(v)) for key, v in baseline.items()}
    global_mean = float(np.mean(y["train"]))
    def baseline_predict(values):
        return np.array([baseline.get((int(r[1]), int(r[0])), global_mean) for r in values])
    candidates = {
        "Ridge (one-hot, alpha=10)": make_pipeline(OneHotEncoder(handle_unknown="ignore", sparse_output=False), Ridge(alpha=10)),
        "Gradient Boosting (150, depth=3)": GradientBoostingRegressor(n_estimators=150, max_depth=3, min_samples_leaf=15, learning_rate=.07, random_state=42),
        "Gradient Boosting (200, depth=4)": GradientBoostingRegressor(n_estimators=200, max_depth=4, min_samples_leaf=15, learning_rate=.07, random_state=42),
        "Histogram Boosting (150, leaves=15)": HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=5, early_stopping=False, random_state=42),
    }
    validation = [{"name": "Hour-of-week baseline", **metrics(y["validation"], baseline_predict(x["validation"]))}]
    for name, model in candidates.items():
        with threadpool_limits(limits=2):
            model.fit(x["train"], y["train"])
            score = metrics(y["validation"], model.predict(x["validation"]))
        validation.append({"name": name, **score})
        print(name, score, flush=True)
    winner = min(validation[1:], key=lambda r: r["mae"])
    model = candidates[winner["name"]]
    # Fixed selection is complete before looking at test labels for evaluation.
    pred = np.maximum(0, model.predict(x["test"]))
    base_test = baseline_predict(x["test"])
    export = export_model(model)
    export.update({"feature_order": FEATURES, "name": winner["name"], "training_period": split["train"]})
    all_x = np.vstack(list(x.values()))
    expected = np.maximum(0, model.predict(all_x))
    actual = np.array([predict_export(export, row) for row in all_x])
    assert np.allclose(actual, expected, rtol=1e-9, atol=1e-7), float(np.max(np.abs(actual-expected)))
    selected = {"name": winner["name"], **metrics(y["test"], pred)}
    base_metrics = {"name": "Hour-of-week baseline", **metrics(y["test"], base_test)}
    improvement = (base_metrics["mae"] - selected["mae"]) / base_metrics["mae"] * 100
    by_hour = [{"hour": hour, "rows": int(np.sum(x["test"][:, 0] == hour)), **metrics(y["test"][x["test"][:, 0] == hour], pred[x["test"][:, 0] == hour])} for hour in range(24)]
    sample_indices = np.linspace(0, len(pred)-1, 72).astype(int)
    examples = [{"timestamp": parts["test"][i]["dteday"] + f'T{int(parts["test"][i]["hr"]):02d}:00', "actual": int(y["test"][i]), "prediction": float(pred[i])} for i in sample_indices]
    results = {"selected": winner["name"], "features": FEATURES, "validation": validation, "test": {"selected": selected, "baseline": base_metrics, "mae_improvement_percent": improvement, "by_hour": by_hour, "examples": examples}, "splits": split, "sklearn_version": sklearn.__version__, "numpy_version": np.__version__, "seed": 42, "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "trained_utc": datetime.now(timezone.utc).isoformat(), "export_max_abs_error": float(np.max(np.abs(actual-expected))), "model_fit_scope": "train only; no refit on validation", "selection": "lowest validation MAE among four predeclared ML candidates; baseline reported as reference", "limitations": ["Historical 2011–2012 demand, not a current-city forecast", "Weather category is supplied, not forecast", "No temperature, wind or humidity inputs", "A single temporal holdout; seasonal and year distribution shift", "Rare severe-weather category has only three training rows", "Training includes 2012 only through May 25; later 2012 seasons require extrapolation"]}
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "browser-model.json").write_text(json.dumps(export, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    joblib.dump(model, ROOT / "artifacts" / "demand-model.joblib", compress=3)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    # A parity fixture covers every calendar/weather combination, not just 72 examples.
    probe = np.array([[h,d,m,w,holiday,int(d not in (0,6) and not holiday),year] for h in range(24) for d in range(7) for m in range(1,13) for w in range(1,5) for holiday in (0,1) for year in (0,1)])
    fixture = {"x": probe.tolist(), "expected": np.maximum(0, model.predict(probe)).tolist()}
    (ROOT / "artifacts" / "parity-fixture.json").write_text(json.dumps(fixture, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"selected": selected, "baseline": base_metrics, "improvement_percent": improvement, "export_max_abs_error": results["export_max_abs_error"]}, indent=2))


if __name__ == "__main__":
    main()
