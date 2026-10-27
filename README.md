# 无监督甲状腺疾病判断

北航《数据挖掘导论》2026 课程大作业任务四。项目基于 6 个脱敏数值特征，仅使用正常样本建立近邻距离异常检测模型，输出异常分数与预测标签：`0` 表示正常，`1` 表示患病。

项目包含数据检查、模型训练、阈值校准、效果评价和结果可视化，并提供基线实验与探索性参数搜索两个入口。

## 方法

采用 kNN 距离异常检测：计算样本与正常训练参考集合中最近的 `k` 个邻居的平均距离，距离越大，异常程度越高。

基线实验流程如下：

1. 将正常训练数据按 80% / 20% 划分为训练子集和验证集，随机种子固定为 `42`。
2. 仅使用训练子集拟合 `StandardScaler`，并建立近邻参考集合。
3. 使用 `k=5` 个邻居的平均欧氏距离作为异常分数。
4. 以正常验证分数的 `0.95` 分位数作为阈值，分位数计算采用 `linear` 方法。
5. 将分数严格大于阈值的样本预测为患病；校准后不再重新拟合模型。

基线的测试标签仅用于评价。探索性搜索会依据测试集指标选择参数，其结果与基线分开报告。

## 数据

将课程提供的 CSV 放入 `dataset/` 目录：

| 文件 | 样本数 | 字段 | 用途 |
| --- | ---: | --- | --- |
| `train-set.csv` | 1,839 | `feature_1` 至 `feature_6` | 正常样本训练与阈值校准 |
| `test-set.csv` | 1,933 | 六个特征及 `label` | 效果评价 |

测试集包含 1,839 个正常样本和 94 个患病样本。训练内部划分为 1,471 个训练样本、368 个验证样本。数据文件不纳入版本管理，需自行准备。

脚本会检查字段、数值有效性、常量列和重复记录。缺列、非法标签、缺失值、非有限值或样本不足时会报错；原始 CSV 不会被修改。

## 环境与运行

实验使用 Python 3.12.3，在 CPU 上运行。主要依赖为 NumPy、pandas、scikit-learn、Matplotlib 和 joblib；`requirements.txt` 记录实验环境的完整依赖版本。

在项目根目录准备环境（以下为 Linux / macOS 命令）：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

### 基线实验

```bash
python run_experiment.py --output-dir results/baseline_20261027
```

默认读取项目目录下的 `dataset/train-set.csv` 和 `dataset/test-set.csv`。如使用其他数据位置，可显式指定：

```bash
python run_experiment.py \
  --train-path /path/to/train-set.csv \
  --test-path /path/to/test-set.csv \
  --output-dir results/baseline_custom
```

输出目录必须尚不存在。重复运行时更换 `--output-dir`，或省略该参数，自动创建带时间戳的目录。显式传入的相对路径按当前工作目录解析；默认路径按脚本所在目录解析。

### 探索性参数搜索

先运行基线，再执行：

```bash
python tune_experiment.py \
  --baseline-dir results/baseline_20261027 \
  --output-dir results/exploratory_20261027
```

搜索沿用基线的划分索引，并核对输入数据哈希和基线预测。数据位于其他位置时，同样使用 `--train-path` 和 `--test-path` 指定。

搜索共包含 308 组配置：

| 参数 | 候选值 |
| --- | --- |
| 缩放方式 | StandardScaler、RobustScaler |
| 距离 | 欧氏距离、曼哈顿距离 |
| 邻居数 `k` | 1、3、5、10、20、30、50 |
| 阈值分位数 `q` | 0.90、0.925、0.95、0.96、0.97、0.975、0.98、0.985、0.99、0.995、0.999 |

分别保存 F1 最高的配置，以及召回率不低于 90% 时 F1 最高的配置。同 F1 时依次优先召回率、AP、较小候选编号。所有配置仅用正常训练子集拟合缩放器和近邻模型，阈值数值由正常验证分数计算。

**搜索使用测试标签选择模型参数及分位数，因此搜索结果仅用于探索性分析，不能视为独立测试性能。**

## 实验结果

以下为 2026 年 10 月 27 日的实验结果。正类固定为患病类别 `1`；AUPRC 使用 PR 曲线梯形积分，AP 使用 Average Precision，二者分别报告。

| 方案 | Precision | Recall | F1 | AUROC | AUPRC | AP | 误报 FP | 漏报 FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 基线 | 0.435644 | 0.936170 | 0.594595 | 0.976687 | 0.801863 | 0.803026 | 114 | 6 |
| 探索性：最高 F1 | 0.736364 | 0.861702 | 0.794118 | 0.978775 | 0.824176 | 0.825269 | 29 | 13 |
| 探索性：Recall ≥ 90% | 0.685484 | 0.904255 | 0.779817 | 0.979782 | 0.831510 | 0.832568 | 39 | 9 |

对应参数如下，阈值展示值保留六位小数，完整精度保存在模型和结果文件中：

| 方案 | 缩放方式 | 距离 | `k` | `q` | 阈值 |
| --- | --- | --- | ---: | ---: | ---: |
| 基线 | StandardScaler | 欧氏 | 5 | 0.95 | 1.430039 |
| 探索性：最高 F1 | RobustScaler | 曼哈顿 | 3 | 0.995 | 5.278288 |
| 探索性：Recall ≥ 90% | RobustScaler | 曼哈顿 | 10 | 0.995 | 6.079305 |

基线检出 94 个患病样本中的 88 个，但误报较多。最高 F1 配置将误报从 114 个降至 29 个，同时漏报从 6 个增至 13 个。召回率约束配置检出 85 个患病样本、误报 39 个，体现了减少误报与保留召回率之间的取舍。

## 输出文件

每次实验写入独立的 `results/` 子目录，已有目录不会被覆盖。结果文件不纳入版本管理，运行脚本后生成。

| 文件或目录 | 内容 |
| --- | --- |
| `config.json`、`environment.json` | 参数、命令、代码与数据哈希、依赖版本 |
| `split_indices.json`、`data_checks.json` | 数据划分、重复与重叠统计 |
| `model.joblib` | 缩放器、近邻模型、特征顺序和阈值 |
| `validation_scores.csv`、`predictions.csv` | 验证分数、逐条测试预测及原始行索引 |
| `metrics.csv`、`summary.json` | 基线指标、混淆矩阵计数和实验摘要 |
| `figures/` | 混淆矩阵、分数分布、ROC 和 PR 图 |

行索引从 `0` 开始，不包含 CSV 表头。运行目录创建后的失败会记录到 `failure.json`。

探索性搜索另外生成 `search_results.csv`（全部候选结果）、`scores/`（各模型分数）和 `comparison.csv`（方案对比）。两套选中模型及各自的预测、摘要和图表分别保存在 `best_f1/` 与 `recall90/`。

加载已保存的模型进行评分，无需重新拟合：

```python
import joblib
import pandas as pd
from run_experiment import predict_labels, score_samples

model = joblib.load("results/baseline_20261027/model.joblib")
samples = pd.read_csv("dataset/test-set.csv")
scores = score_samples(model, samples)
predictions = predict_labels(scores, model["threshold"])
```

## 局限与复现说明

- 数据存在重复特征：训练集和测试集分别有 57、23 条重复记录；训练与测试共享 36 种特征向量。实验保留原始记录并按行划分，重叠可能使评价偏乐观。
- 六个特征均已脱敏，上游预处理及患者独立性未知，相同特征不能证明来自同一患者。
- 验证集只有正常样本。较高分位数依赖少量尾部样本，阈值稳定性需要进一步验证。
- 探索性配置依据重复使用的测试集选择，其提升需要新的、未参与参数选择的带标签数据验证。

已核验划分互斥、缩放器拟合范围、近邻距离计算、阈值边界、指标重算及模型重载后的预测一致性。基线与探索性搜索使用相同数据和划分，完整参数、索引及环境版本随实验结果保存。
