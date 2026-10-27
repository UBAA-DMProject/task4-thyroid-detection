# 无监督甲状腺疾病判断

北航《数据挖掘导论》2026 课程大作业任务四：基于 6 个脱敏数值特征，仅使用正常样本训练，输出异常分数及预测标签（`0` 为正常，`1` 为患病）。本次选择近邻距离异常检测（kNN 距离法）作为主方法；Isolation Forest 和 One-Class SVM 仅作为可选扩展，不属于当前必做范围。

当前已准备数据与实验方案，训练、评估和绘图代码尚未实现。

## 环境与配置

- 计划环境：Python 3、NumPy、pandas、scikit-learn、Matplotlib；依赖版本待实现时确定。
- 训练数据：`dataset/train-set.csv`，仅含正常样本。
- 测试数据：`dataset/test-set.csv`，`label` 仅用于评价。
- 方法：标准化后查询正常训练子集中最近的 5 个邻居，以平均欧氏距离作为异常分数；不使用标签投票的 kNN 分类器。
- 初始实验约定：训练内部按 80% / 20% 划分，随机种子为 `42`，使用 `StandardScaler`，阈值取正常验证分数的 `0.95` 分位数。

## 启动

暂未提供依赖清单和运行入口。实现后将在此补充经过验证的安装、训练与评估命令。

## 项目结构

```text
task4-thyroid-detection/
├── README.md                # 项目说明
├── AGENTS.md                # 协作与实现约定
├── dataset/                 # 原始数据（不纳入版本管理）
│   ├── train-set.csv
│   └── test-set.csv
└── docs/
    ├── 实验方案.md          # 数据说明、模型方案、评价规则与实现规划
    └── 任务四报告初稿.md    # 问题定义、方法选择与理由
```

## 相关文档

- [实验方案](docs/实验方案.md)
- [报告初稿](docs/任务四报告初稿.md)
- [课程要求](../course-project/README.md)
- [正式作业说明](../course-project/docs/作业说明_2026.pdf)
