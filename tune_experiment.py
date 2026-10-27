"""根据已使用的测试集探索 kNN 参数；结果不是独立测试性能。"""

import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
from itertools import product
import json
from pathlib import Path
import platform
import shlex
import sys
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from run_experiment import (
    FEATURES, ROOT, evaluate, fit_model, overlap_report, plot_results,
    predict_labels, read_data, score_samples, write_json,
)


GRID = {
    "scaler": ["standard", "robust"],
    "metric": ["euclidean", "manhattan"],
    "neighbors": [1, 3, 5, 10, 20, 30, 50],
    "quantile": [0.90, 0.925, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 0.999],
}
EXPLORATORY_NOTE = "超参数及分位数依据重复使用的测试标签选择；仅为探索性结果，不是独立泛化性能。"


def select_candidate(candidates, minimum_recall=0.0):
    eligible = [row for row in candidates if row["recall"] >= minimum_recall]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (row["f1"], row["recall"], row["ap"], -row["candidate_id"]))


def verify_split(split, size):
    training = np.asarray(split["train"])
    validation = np.asarray(split["validation"])
    if any(indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) for indices in [training, validation]):
        raise ValueError("基线划分必须为一维整数索引")
    if len(training) < max(GRID["neighbors"]) or len(validation) == 0:
        raise ValueError("基线划分样本不足以执行搜索")
    if not np.array_equal(np.sort(np.concatenate([training, validation])), np.arange(size)):
        raise ValueError("基线划分必须无重复、互斥且完整覆盖训练集")
    return training, validation


def save_selection(output, candidate, model, validation, test, validation_indices):
    output.mkdir()
    model = {**model, "threshold": candidate["threshold"], "selection": candidate,
             "evaluation_status": "exploratory_test_tuned", "limitations": EXPLORATORY_NOTE}
    validation_scores = score_samples(model, validation)
    scores = score_samples(model, test)
    predictions = predict_labels(scores, model["threshold"])
    labels = test["label"].to_numpy(dtype=int)
    joblib.dump(model, output / "model.joblib")
    write_json(output / "summary.json", {**candidate, "evaluation_status": "exploratory_test_tuned",
                                        "limitations": EXPLORATORY_NOTE})
    pd.DataFrame({"row_index": validation_indices, "score": validation_scores,
                  "threshold": model["threshold"],
                  "prediction": predict_labels(validation_scores, model["threshold"])}).to_csv(output / "validation_scores.csv", index=False)
    pd.DataFrame({"row_index": np.arange(len(test)), "model": candidate["model_id"], "label": labels,
                  "score": scores, "threshold": model["threshold"],
                  "prediction": predictions}).to_csv(output / "predictions.csv", index=False)
    plot_results(output / "figures", labels, scores, predictions, model["threshold"])


def run_search(args):
    started = perf_counter()
    output = args.output_dir.resolve()
    baseline_dir = args.baseline_dir.resolve()
    for source in [args.train_path.resolve(), args.test_path.resolve()]:
        if output == source.parent or source.parent in output.parents:
            raise ValueError("结果目录不能位于原始数据目录内")
    if output == baseline_dir or baseline_dir in output.parents:
        raise ValueError("调参结果必须保存到独立于基线的目录")
    output.mkdir(parents=True, exist_ok=False)
    try:
        baseline_config = json.loads((baseline_dir / "config.json").read_text())
        config = {
            "evaluation_status": "exploratory_test_tuned", "limitations": EXPLORATORY_NOTE,
            "grid": GRID, "selection_rule": "F1 最大；同分依次优先 Recall、AP、较小候选编号",
            "alternative_rule": "Recall >= 0.90 后使用同一排序规则",
            "quantile_method": "linear", "decision_rule": "score > threshold",
            "fit_data": "仅正常训练子集；不重训、不改种子、不去重",
            "threshold_source": "正常验证分数的分位数；分位数参数由测试集指标选择",
            "baseline_dir": str(baseline_dir), "features": FEATURES,
            "created_at": datetime.now().astimezone().isoformat(),
            "command": shlex.join([sys.executable, *sys.argv]), "working_directory": str(Path.cwd()),
            "script_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in ["run_experiment.py", "tune_experiment.py"]},
            "inputs": {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                       for name, path in [("train", args.train_path), ("test", args.test_path)]},
        }
        write_json(output / "config.json", config)
        for name in ["train", "test"]:
            if config["inputs"][name]["sha256"] != baseline_config["inputs"][name]["sha256"]:
                raise ValueError(f"{name} 数据与基线不一致")
        write_json(output / "environment.json", {
            "python": platform.python_version(), "executable": sys.executable,
            "packages": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "joblib", "threadpoolctl"]},
        })
        train, train_report = read_data(args.train_path)
        test, test_report = read_data(args.test_path, require_label=True)
        labels = test["label"].to_numpy(dtype=int)
        if len(np.unique(labels)) != 2:
            raise ValueError("探索性搜索需要同时包含两类标签")
        split = json.loads((baseline_dir / "split_indices.json").read_text())
        train_indices, validation_indices = verify_split(split, len(train))
        write_json(output / "split_indices.json", split)
        training, validation = train.iloc[train_indices], train.iloc[validation_indices]
        write_json(output / "data_checks.json", {
            "train": train_report, "test": test_report,
            "train_test_overlap": overlap_report(train, test),
            "training_validation_overlap": overlap_report(training, validation),
            "duplicate_policy": "保留原始记录及基线划分，重复特征可能使结果偏乐观。",
        })
        baseline_predictions = pd.read_csv(baseline_dir / "predictions.csv")
        baseline_model = fit_model(training)
        baseline_scores = score_samples(baseline_model, test)
        baseline_threshold = float(np.quantile(score_samples(baseline_model, validation), 0.95, method="linear"))
        baseline_labels = predict_labels(baseline_scores, baseline_threshold)
        np.testing.assert_allclose(baseline_scores, baseline_predictions["score"], rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(baseline_labels, baseline_predictions["prediction"])
        baseline_metrics, _ = evaluate(labels, baseline_scores, baseline_labels)
        write_json(output / "baseline_metrics.json", baseline_metrics)
        models = {}
        candidates = []
        scores_dir = output / "scores"
        scores_dir.mkdir()
        for scaler_name, metric, neighbors in product(GRID["scaler"], GRID["metric"], GRID["neighbors"]):
            model_id = f"{scaler_name}_{metric}_k{neighbors}"
            model = fit_model(training, n_neighbors=neighbors, metric=metric, scaler_name=scaler_name)
            models[model_id] = model
            validation_scores = score_samples(model, validation)
            scores = score_samples(model, test)
            pd.DataFrame({"row_index": validation_indices, "score": validation_scores}).to_csv(scores_dir / f"{model_id}_validation.csv", index=False)
            pd.DataFrame({"row_index": np.arange(len(test)), "label": labels, "score": scores}).to_csv(scores_dir / f"{model_id}_test.csv", index=False)
            for quantile in GRID["quantile"]:
                threshold = float(np.quantile(validation_scores, quantile, method="linear"))
                predictions = predict_labels(scores, threshold)
                metrics, notes = evaluate(labels, scores, predictions)
                candidates.append({
                    "candidate_id": len(candidates), "model_id": model_id, "scaler": scaler_name,
                    "metric": metric, "neighbors": neighbors, "quantile": quantile, "threshold": threshold,
                    "validation_false_positive_rate": float(np.mean(validation_scores > threshold)),
                    **metrics, "notes": "；".join(notes), "evaluation_status": "exploratory_test_tuned",
                })
            pd.DataFrame(candidates).to_csv(output / "search_results.csv", index=False)
        best = select_candidate(candidates)
        recall90 = select_candidate(candidates, minimum_recall=0.90)
        selected = {"best_f1": best}
        if recall90 is not None:
            selected["recall90"] = recall90
        for name, candidate in selected.items():
            save_selection(output / name, candidate, models[candidate["model_id"]], validation, test, validation_indices)
        comparison = [{"selection": "original_baseline", **baseline_metrics}]
        comparison.extend({"selection": name, **candidate} for name, candidate in selected.items())
        pd.DataFrame(comparison).to_csv(output / "comparison.csv", index=False)
        summary = {
            "status": "complete", "evaluation_status": "exploratory_test_tuned", "limitations": EXPLORATORY_NOTE,
            "candidate_count": len(candidates), "fitted_models": len(models), "baseline_metrics": baseline_metrics,
            "selected": selected, "recall90_available": recall90 is not None,
            "total_seconds": perf_counter() - started,
        }
        write_json(output / "summary.json", summary)
        print(json.dumps({"output_dir": str(output), **summary}, ensure_ascii=False, indent=2))
    except Exception as error:
        write_json(output / "failure.json", {"status": "failed", "error": str(error), "type": type(error).__name__})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=ROOT / "dataset/train-set.csv")
    parser.add_argument("--test-path", type=Path, default=ROOT / "dataset/test-set.csv")
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "results/baseline_20261027")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / datetime.now().strftime("exploratory_%Y%m%d_%H%M%S_%f"))
    args = parser.parse_args()
    try:
        run_search(args)
    except (ValueError, OSError, AssertionError, KeyError) as error:
        parser.exit(1, f"调参失败：{error}\n")


if __name__ == "__main__":
    main()
