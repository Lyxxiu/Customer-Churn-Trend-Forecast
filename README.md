# 电信客户流失预测 —— 可复现代码

基于监督学习的大数据分析演示，目标是**识别可能流失（Churn）的客户**。
完整实现 CRISP-DM 数据挖掘流程：业务理解 → 数据理解（EDA）→ 数据准备 → 特征工程 → 建模 → 评估 → 业务建议。

## 本文件夹内容

```
客户流失分析-学生版/
├── churn_analysis.py          # 完整分析代码（运行这个）
├── Telco_Customer_Churn.csv   # 数据（7032 行 × 19 列）
├── requirements.txt           # 依赖清单
└── README.md                  # 本说明
```

## 一、环境要求

- Python 3.9 及以上（推荐 3.10+）
- Windows / macOS / Linux 均可
- Windows 系统默认自带微软雅黑字体，图表中文正常显示

## 二、安装依赖

打开命令行（Windows 用 cmd 或 PowerShell），进入本文件夹，执行：

```bash
pip install -r requirements.txt
```

> 提示：如果 pip 下载慢，可加国内镜像：
> `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

## 三、运行

```bash
python churn_analysis.py
```

整个流程约需 **2～5 分钟**（其中网格搜索调参最耗时，属正常现象）。

## 四、输出结果

运行结束后，本文件夹会自动生成：

1. `figures/` —— 10 张图表：
   - `01_class_balance.png` 目标变量（流失）分布
   - `02_churn_by_features.png` 各特征对流失方向的影响
   - `03_charges_dist.png` 月费 / 累计费用分布
   - `04_correlation.png` 特征相关性热力图
   - `05_roc_curves.png` 三模型 ROC 曲线
   - `06_pr_curves.png` 精确率-召回率曲线
   - `07_confusion_matrix.png` 最优模型混淆矩阵
   - `08_feature_importance.png` 随机森林特征重要性
   - `09_permutation_importance.png` 排列重要性
   - `10_engineered_features_effect.png` 特征工程收益对比
2. `metrics.json` —— 全部关键指标（AUC、F1、召回率、混淆矩阵等）

## 五、预期结果（参考）

| 项目 | 结果 |
|---|---|
| 数据规模 | 7032 行 × 19 列，去重后 6982 行 |
| 流失率 | 26.58%（类别不平衡） |
| 特征工程 | 18 个原始特征 → 36 个特征（新增 18 个派生特征） |
| 最优模型 | 逻辑回归 |
| 测试 AUC | ≈ 0.8366 |
| 召回率 Recall | ≈ 0.756（抓住约 75.6% 的流失客户） |

> 因固定了随机种子 `random_state=42`，同一版本环境下结果完全一致。

## 六、常见问题

1. **报错找不到数据文件**：确认 `Telco_Customer_Churn.csv` 与 `churn_analysis.py` 在同一目录。
2. **图表中文乱码 / 字体警告**（Mac、Linux）：把脚本中两处 `Microsoft YaHei` 改成 `SimHei` 或 `PingFang SC`。
3. **运行很慢**：网格搜索（GridSearchCV）是主要耗时点，2～5 分钟属正常；想更快可把 `RandomForest` 的 `n_estimators` 调小。
4. **提示缺模块**：先确认执行过 `pip install -r requirements.txt`。

## 七、代码结构速览

| 模块 | 作用 |
|---|---|
| 0 业务理解 | 明确目标与评估指标 |
| 1 数据理解 | 读数据、数据字典、4 张 EDA 图 |
| 2 数据准备 | 缺失/重复检查、去重 |
| 3 特征工程 | `engineer_features()` 构造 18 个派生特征 |
| 4 数据划分 | 分层抽样 80/20 |
| 5 建模 | `build_model()`：标准化 + SMOTE + 网格搜索 |
| 6 评估 | ROC / PR / 混淆矩阵 / 分类报告 |
| 7 解释 | 特征重要性 + 排列重要性 |
| 8 对比 | 原始特征 vs 完整特征的 AUC 提升 |
| 9 结论 | 业务建议 + 写 metrics.json |
