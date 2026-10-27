# 数据放置说明

请将课程提供的甲状腺疾病数据文件 **`train-set.csv`** 和 **`test-set.csv`** 放在本目录，保持文件名和原始内容不变：

```text
dataset/
├── README.md
├── train-set.csv
└── test-set.csv
```

训练集应包含 `feature_1` 至 `feature_6` 六个脱敏数值特征，仅含正常样本；测试集应包含相同的六个特征及 `label` 列，其中 `0` 表示正常，`1` 表示患病。程序按上述顺序选择特征，`label` 仅用于评价，不进入模型输入。

原始 CSV 不随仓库提供，需自行从课程数据中获取。本目录除本说明外的内容均被 Git 忽略；实验结果写入 `results/`，不写入本目录。安装与运行方法见 [项目 README](../README.md)。
