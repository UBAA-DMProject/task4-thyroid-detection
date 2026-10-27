"""使用正常样本训练、校准并评价 kNN 距离异常检测模型。"""

import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shlex
import sys
import tempfile
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler


ROOT = Path(__file__).resolve().parent
FEATURES = [f"feature_{number}" for number in range(1, 7)]


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_data(path, require_label=False):
    frame = pd.read_csv(path)
    required = FEATURES + (["label"] if require_label else [])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: 缺少列 {missing}")
    if frame.empty:
        raise ValueError(f"{path}: 数据为空")
    try:
        values = frame[FEATURES].to_numpy(dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{path}: 特征必须为数值") from error
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: 特征包含缺失值或非有限值")
    frame[FEATURES] = values
    if "label" in frame:
        if not frame["label"].isin([0, 1]).all():
            raise ValueError(f"{path}: label 必须为 0 或 1")
        if not require_label and frame["label"].ne(0).any():
            raise ValueError(f"{path}: 训练数据只能包含正常样本")
    report = {
        "rows": len(frame),
        "missing_values": 0,
        "nonfinite_values": 0,
        "constant_columns": [name for name in FEATURES if frame[name].nunique() == 1],
        "duplicate_feature_rows": int(frame.duplicated(FEATURES).sum()),
    }
    return frame, report


def overlap_report(first, second):
    first_counts = first.groupby(FEATURES).size().rename("first_count")
    second_counts = second.groupby(FEATURES).size().rename("second_count")
    shared = pd.concat([first_counts, second_counts], axis=1, join="inner")
    return {
        "unique_feature_vectors": len(shared),
        "matching_pairs": int((shared.first_count * shared.second_count).sum()),
        "first_rows": int(shared.first_count.sum()),
        "second_rows": int(shared.second_count.sum()),
    }


def fit_model(training, n_neighbors=5, metric="euclidean", scaler_name="standard"):
    if not isinstance(n_neighbors, int) or n_neighbors < 1 or len(training) < n_neighbors:
        raise ValueError("近邻数量必须为正整数且不超过正常训练子集大小")
    if scaler_name not in {"standard", "robust"}:
        raise ValueError("缩放方式必须为 standard 或 robust")
    if metric not in {"euclidean", "manhattan"}:
        raise ValueError("距离必须为 euclidean 或 manhattan")
    scaler = StandardScaler() if scaler_name == "standard" else RobustScaler()
    scaled = scaler.fit_transform(training[FEATURES])
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm="brute", n_jobs=1)
    neighbors.fit(scaled)
    return {"scaler": scaler, "neighbors": neighbors, "features": FEATURES.copy()}


def score_samples(model, frame):
    scaled = model["scaler"].transform(frame[model["features"]])
    distances = model["neighbors"].kneighbors(scaled, return_distance=True)[0]
    scores = distances.mean(axis=1)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("模型未生成有限的一维异常分数")
    return scores


def predict_labels(scores, threshold):
    return (np.asarray(scores) > threshold).astype(int)


def evaluate(labels, scores, predictions):
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    both_classes = len(np.unique(labels)) == 2
    notes = []
    if both_classes:
        precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
        ranking = {
            "auroc": float(roc_auc_score(labels, scores)),
            "auprc": float(auc(recall, precision)),
            "ap": float(average_precision_score(labels, scores)),
        }
    else:
        ranking = dict.fromkeys(["auroc", "auprc", "ap"])
        notes.append("测试集仅包含一个类别，排序指标不报告，ROC/PR 图跳过。")
    if not np.any(predictions == 1):
        notes.append("没有预测正类；分类指标使用 zero_division=0。")
    metrics = {
        **ranking,
        "precision": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1]),
    }
    return metrics, notes


def plot_results(directory, labels, scores, predictions, threshold):
    with tempfile.TemporaryDirectory(prefix="thyroid-mpl-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", cache)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        directory.mkdir()
        figure, axis = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(labels, predictions, labels=[0, 1], ax=axis, colorbar=False)
        figure.savefig(directory / "confusion_matrix.png", dpi=160, bbox_inches="tight")
        plt.close(figure)
        figure, axis = plt.subplots()
        bins = np.histogram_bin_edges(scores, bins=40)
        for label, name in [(0, "Normal"), (1, "Disease")]:
            axis.hist(scores[labels == label], bins=bins, alpha=0.6, label=name)
        axis.axvline(threshold, color="black", linestyle="--", label="Threshold")
        axis.set(xlabel="Mean neighbor distance", ylabel="Sample count", yscale="symlog")
        axis.legend()
        figure.savefig(directory / "score_distribution.png", dpi=160, bbox_inches="tight")
        plt.close(figure)
        if len(np.unique(labels)) == 2:
            false_positive, true_positive, _ = roc_curve(labels, scores, pos_label=1)
            precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
            for name, horizontal, vertical, xlabel, ylabel in [
                ("roc", false_positive, true_positive, "False positive rate", "True positive rate"),
                ("pr", recall, precision, "Recall", "Precision"),
            ]:
                figure, axis = plt.subplots()
                axis.plot(horizontal, vertical)
                axis.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.02))
                figure.savefig(directory / f"{name}.png", dpi=160, bbox_inches="tight")
                plt.close(figure)


def run_experiment(args):
    started = perf_counter()
    output = args.output_dir.resolve()
    for source in [args.train_path.resolve(), args.test_path.resolve()]:
        if output == source.parent or source.parent in output.parents:
            raise ValueError("结果目录不能位于原始数据目录内")
    output.mkdir(parents=True, exist_ok=False)
    try:
        config = {
            "model": "knn_mean_distance", "features": FEATURES, "seed": 42,
            "validation_fraction": 0.2, "neighbors": 5, "metric": "euclidean",
            "algorithm": "brute", "n_jobs": 1, "scaler": "StandardScaler",
            "quantile": 0.95, "quantile_method": "linear", "decision_rule": "score > threshold",
            "duplicate_policy": "保留重复记录，按原始行随机划分",
            "created_at": datetime.now().astimezone().isoformat(),
            "command": shlex.join([sys.executable, *sys.argv]),
            "working_directory": str(Path.cwd()),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "inputs": {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                       for name, path in [("train", args.train_path), ("test", args.test_path)]},
        }
        write_json(output / "config.json", config)
        write_json(output / "environment.json", {
            "python": platform.python_version(), "executable": sys.executable,
            "packages": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "joblib", "threadpoolctl"]},
        })
        train, train_report = read_data(args.train_path)
        test, test_report = read_data(args.test_path, require_label=True)
        if len(train) < 7:
            raise ValueError("训练集至少需要 7 条样本，以支持 80%/20% 划分和 5 个近邻")
        train_indices, validation_indices = train_test_split(np.arange(len(train)), test_size=0.2, random_state=42)
        training, validation = train.iloc[train_indices], train.iloc[validation_indices]
        write_json(output / "split_indices.json", {"train": train_indices.tolist(), "validation": validation_indices.tolist()})
        write_json(output / "data_checks.json", {
            "train": train_report, "test": test_report,
            "training_constant_columns": [name for name in FEATURES if training[name].nunique() == 1],
            "train_test_overlap": overlap_report(train, test),
            "training_validation_overlap": overlap_report(training, validation),
            "constant_policy": "保留常量列；StandardScaler 对零方差列使用单位缩放。",
            "duplicate_policy": config["duplicate_policy"],
        })
        fit_started = perf_counter()
        model = fit_model(training)
        fit_seconds = perf_counter() - fit_started
        validation_scores = score_samples(model, validation)
        threshold = float(np.quantile(validation_scores, 0.95, method="linear"))
        model["threshold"] = threshold
        joblib.dump(model, output / "model.joblib")
        pd.DataFrame({"row_index": validation_indices, "score": validation_scores,
                      "threshold": threshold, "prediction": predict_labels(validation_scores, threshold)}).to_csv(output / "validation_scores.csv", index=False)
        scores = score_samples(model, test)
        predictions = predict_labels(scores, threshold)
        labels = test["label"].to_numpy(dtype=int)
        metrics, notes = evaluate(labels, scores, predictions)
        pd.DataFrame([{"model": config["model"], **metrics}]).to_csv(output / "metrics.csv", index=False)
        pd.DataFrame({"row_index": np.arange(len(test)), "model": config["model"], "label": labels,
                      "score": scores, "threshold": threshold, "prediction": predictions}).to_csv(output / "predictions.csv", index=False)
        plot_results(output / "figures", labels, scores, predictions, threshold)
        summary = {
            "status": "complete", "threshold": threshold,
            "validation_false_positive_rate": float(np.mean(validation_scores > threshold)),
            "training_rows": len(training), "validation_rows": len(validation), "test_rows": len(test),
            "metrics": metrics, "notes": notes,
            "limitations": ["重复特征可能使验证及测试结果偏乐观；未去重或改变划分。",
                            "特征脱敏，上游预处理及患者独立性未知；无患病验证样本。",
                            "仅运行预先约定的基线，未使用测试结果调参。"],
            "fit_seconds": fit_seconds, "total_seconds": perf_counter() - started,
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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / datetime.now().strftime("baseline_%Y%m%d_%H%M%S_%f"))
    args = parser.parse_args()
    try:
        run_experiment(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f"实验失败：{error}\n")


if __name__ == "__main__":
    main()
