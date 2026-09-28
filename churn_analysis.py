# -*- coding: utf-8 -*-
"""
电信客户流失预测 —— 完整监督学习大数据分析演示（学生可复现版）
=============================================================
数据集：Telco_Customer_Churn.csv（7032 行 × 19 列，与本脚本同目录）
目标  ：识别可能流失（Churn）的客户

分析流程（参照 CRISP-DM 数据挖掘标准流程）：
  0. 业务理解  —— 明确目标、评估标准
  1. 数据理解  —— 数据字典、EDA、可视化
  2. 数据准备  —— 清洗、去重
  3. 特征工程  —— 构造派生特征、交互特征（本演示重点）
  4. 建模      —— 逻辑回归 / 随机森林 / 梯度提升
  5. 评估      —— 交叉验证、ROC/PR、混淆矩阵、特征重要性
  6. 部署与结论 —— 业务建议

运行前：
  1) 安装依赖：pip install -r requirements.txt
  2) 运行：     python churn_analysis.py

输出：
  figures/     目录下生成 10 张图表
  metrics.json 记录全部关键指标
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score,
                             classification_report, confusion_matrix,
                             roc_curve, precision_recall_curve)
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 全局配置（可移植：数据与输出都放在本脚本所在目录，无需修改路径）
# ----------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))          # 脚本所在目录
DATA = os.path.join(BASE, "Telco_Customer_Churn.csv")      # 数据文件（同目录）
FIG = os.path.join(BASE, "guizi")                        # 图表输出目录
os.makedirs(FIG, exist_ok=True)

if not os.path.exists(DATA):
    raise SystemExit(
        f"\n找不到数据文件：{DATA}\n"
        f"请确认 Telco_Customer_Churn.csv 与本脚本放在同一目录下。\n")

# 中文字体（Windows 用微软雅黑；Mac/Linux 若报字体警告可改为 SimHei / PingFang SC）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", context="notebook", font="Microsoft YaHei")

NAVY = "#0F2A43"
AMBER = "#F59E0B"
TEAL = "#2E86AB"
GRAY = "#6B7280"
BLUE = "#4F81BD"
GREEN = "#3E9B6C"
RED = "#C0504D"

RANDOM = 42            # 固定随机种子，保证结果可复现

METRICS = {}          # 汇总指标，最终写入 metrics.json
LOG = []              # 关键日志，便于引用


def log(msg):
    print(msg)
    LOG.append(msg)


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=160, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  [图] saved", name)


# ======================================================================
# 0. 业务理解
# ======================================================================
log("=" * 70)
log("0. 业务理解")
log("=" * 70)
log("目标：预测客户是否会流失（Churn=1），以便提前干预、减少流失损失。")
log("评估指标：ROC-AUC（整体排序能力）+ 召回率 Recall（尽量抓住流失客户）")
log("")

# ======================================================================
# 1. 数据理解
# ======================================================================
log("=" * 70)
log("1. 数据理解（EDA）")
log("=" * 70)
df = pd.read_csv(DATA)
log(f"数据规模：{df.shape[0]} 行 × {df.shape[1]} 列")
log(f"流失率：{df['Churn'].mean()*100:.2f}%（{df['Churn'].sum()} / {len(df)}）")
log("特征均为数值型（分类变量已 0/1 二值化；MonthlyCharges/TotalCharges 为连续）")

# 数据字典（依据业务含义与流失方向推断编码语义）
data_dict = {
    "SeniorCitizen": "是否老年客户（1=是，风险↑）",
    "Partner": "是否有伴侣（1=是，保护↓）",
    "Dependents": "是否有受抚养人（1=是，保护↓）",
    "tenure": "是否老客户（1=是，保护↓，此列已二值化）",
    "PhoneService": "是否开通电话（1=是）",
    "MultipleLines": "是否多线路（1=是）",
    "InternetService": "是否光纤宽带（1=光纤，风险↑）",
    "OnlineSecurity": "是否开通在线安全（1=是，保护↓）",
    "OnlineBackup": "是否在线备份（1=是）",
    "DeviceProtection": "是否设备保护（1=是）",
    "TechSupport": "是否技术支持（1=是，保护↓）",
    "StreamingTV": "是否流媒体电视（1=是）",
    "StreamingMovies": "是否流媒体电影（1=是）",
    "Contract": "是否长期合同（1=长期，保护↓↓）",
    "PaperlessBilling": "是否电子账单（1=是，风险↑）",
    "PaymentMethod": "是否电子支票支付（1=是，风险↑）",
    "MonthlyCharges": "月费（连续）",
    "TotalCharges": "累计总费用（连续）",
    "Churn": "目标：是否流失（1=流失）",
}
log("数据字典已生成（详见 PPT）。")
METRICS["n_rows"] = int(len(df))
METRICS["n_cols"] = int(df.shape[1])
METRICS["churn_rate"] = float(df["Churn"].mean())
METRICS["n_churn"] = int(df["Churn"].sum())
METRICS["n_nochurn"] = int((df["Churn"] == 0).sum())

# ---- 1.1 目标变量分布 ----
fig, ax = plt.subplots(figsize=(6.5, 4))
cnt = df["Churn"].value_counts().sort_index()
bars = ax.bar(["未流失 (0)", "流失 (1)"], cnt.values,
              color=[GREEN, RED], width=0.5)
for b, v in zip(bars, cnt.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 40, f"{v}\n{v/len(df)*100:.1f}%",
            ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_title("目标变量 Churn 的分布（类别不平衡）", fontsize=14, fontweight="bold")
ax.set_ylabel("客户数")
ax.set_ylim(0, cnt.values.max() * 1.2)
savefig(fig, "01_class_balance.png")

# ---- 1.2 各二值特征的流失率 ----
feat_risk = {}
for c in df.columns:
    if c in ("Churn", "MonthlyCharges", "TotalCharges"):
        continue
    g = df.groupby(c)["Churn"].agg(["mean", "count"])
    feat_risk[c] = (g.loc[1, "mean"] - g.loc[0, "mean"])  # 1 相对 0 的流失率差

order = sorted(feat_risk, key=feat_risk.get)
fig, ax = plt.subplots(figsize=(8.5, 6.5))
vals = [feat_risk[c] * 100 for c in order]
colors = [RED if v > 0 else GREEN for v in vals]
ax.barh(order, vals, color=colors)
ax.axvline(0, color=GRAY, lw=1)
ax.set_xlabel("特征取 1 时流失率的相对变化（百分点）")
ax.set_title("各特征对流失方向的影响（正=风险因子，负=保护因子）",
             fontsize=13, fontweight="bold")
for i, v in enumerate(vals):
    ax.text(v + (0.3 if v >= 0 else -0.3), i, f"{v:+.1f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=9)
savefig(fig, "02_churn_by_features.png")

# ---- 1.3 费用分布 ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, col, c in [(axes[0], "MonthlyCharges", TEAL),
                   (axes[1], "TotalCharges", NAVY)]:
    for churn, lab, color in [(0, "未流失", GREEN), (1, "流失", RED)]:
        sub = df.loc[df.Churn == churn, col]
        ax.hist(sub, bins=40, alpha=0.55, label=lab, color=color)
    ax.set_title(col + " 分布（按流失分层）", fontsize=12, fontweight="bold")
    ax.legend()
savefig(fig, "03_charges_dist.png")

# ---- 1.4 相关性热力图 ----
corr = df.corr()
fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.3, ax=ax,
            annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
ax.set_title("特征间相关性热力图", fontsize=14, fontweight="bold")
savefig(fig, "04_correlation.png")

# ======================================================================
# 2. 数据准备（清洗与基础变换）
# ======================================================================
log("=" * 70)
log("2. 数据准备")
log("=" * 70)
# 检查缺失值 / 异常
log(f"缺失值总数：{int(df.isna().sum().sum())}")
log(f"重复行：{int(df.duplicated().sum())}")
log(f"TotalCharges==0：{int((df['TotalCharges']==0).sum())}")
log(f"TotalCharges < MonthlyCharges：{int((df['TotalCharges'] < df['MonthlyCharges']).sum())}")
# 去重（无缺失值、无 0 值异常）
n_dup = int(df.duplicated().sum())
if n_dup > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    log(f"已去除 {n_dup} 行重复记录，剩余 {len(df)} 行")
METRICS["missing_total"] = int(df.isna().sum().sum())
METRICS["duplicates"] = n_dup
METRICS["n_rows_after_dedup"] = int(len(df))

X_raw = df.drop(columns=["Churn"])
y = df["Churn"].astype(int)

# ======================================================================
# 3. 特征工程（重点）
# ======================================================================
log("=" * 70)
log("3. 特征工程（深入）")
log("=" * 70)

service_cols = ["PhoneService", "MultipleLines", "OnlineSecurity",
                "OnlineBackup", "DeviceProtection", "TechSupport",
                "StreamingTV", "StreamingMovies"]
risk_cols = ["SeniorCitizen", "InternetService", "PaymentMethod",
             "PaperlessBilling"]
protect_cols = ["Contract", "Partner", "Dependents", "OnlineSecurity",
                "TechSupport", "tenure"]

X_base = X_raw.copy()  # 原始特征（18 个）


def engineer_features(X):
    """在原始特征基础上构造派生特征（均为 X 的行内函数，无目标泄漏）。"""
    Xe = X.copy()
    # 1) 服务订阅总数：订阅越多，客户粘性/价值越高
    Xe["ServiceCount"] = X[service_cols].sum(axis=1)
    # 2) 家庭承诺度：有伴侣 + 有受抚养人
    Xe["FamilyFactor"] = X["Partner"] + X["Dependents"]
    # 3) 估计在网月数：累计费用 ÷ 月费（近似使用时长）
    Xe["EstimatedMonths"] = X["TotalCharges"] / X["MonthlyCharges"].replace(0, np.nan)
    # 4) 月费占累计费用比：越高说明越接近“新客户/近期高消费”
    Xe["MonthlyToTotal"] = X["MonthlyCharges"] / X["TotalCharges"].replace(0, np.nan)
    # 5) 单服务平均价值：月费 ÷ (服务数+1)
    Xe["ValuePerService"] = X["MonthlyCharges"] / (X[service_cols].sum(axis=1) + 1)
    # 6) 高消费客户（业务阈值：月费 > 70）
    Xe["HighSpender"] = (X["MonthlyCharges"] > 70).astype(int)
    # 7) 对数变换（缓解右偏、缩放量纲）
    Xe["LogTotalCharges"] = np.log1p(X["TotalCharges"])
    Xe["LogMonthlyCharges"] = np.log1p(X["MonthlyCharges"])
    # 8) 交互特征（捕捉组合效应）
    Xe["Contract_x_tenure"] = X["Contract"] * X["tenure"]          # 长期合同 × 老客户
    Xe["Contract_x_Monthly"] = X["Contract"] * X["MonthlyCharges"]  # 长期合同 × 月费
    Xe["tenure_x_Monthly"] = X["tenure"] * X["MonthlyCharges"]
    Xe["Internet_x_Monthly"] = X["InternetService"] * X["MonthlyCharges"]  # 光纤 × 月费
    Xe["Paperless_x_Payment"] = X["PaperlessBilling"] * X["PaymentMethod"]  # 电子账单+电子支付
    Xe["Contract_x_Internet"] = X["Contract"] * X["InternetService"]
    # 9) 忠诚度综合分：长期合同 + 老客户
    Xe["LoyaltyScore"] = X["Contract"] + X["tenure"]
    # 10) 风险因子计数 / 保护因子计数 / 净保护
    Xe["RiskCount"] = X[risk_cols].sum(axis=1)
    Xe["ProtectCount"] = X[protect_cols].sum(axis=1)
    Xe["NetProtection"] = Xe["ProtectCount"] - Xe["RiskCount"]
    return Xe


X_eng = engineer_features(X_raw)
log(f"原始特征：{X_base.shape[1]} 个；特征工程后：{X_eng.shape[1]} 个（新增 {X_eng.shape[1]-X_base.shape[1]} 个）")
log("新增派生特征：")
log("  · ServiceCount / FamilyFactor —— 服务与家庭维度")
log("  · EstimatedMonths / MonthlyToTotal —— 由费用反推时长与消费结构")
log("  · ValuePerService / HighSpender / Log 变换 —— 价值与量纲处理")
log("  · Contract×tenure、Contract×Monthly 等 6 个交互项 —— 组合效应")
log("  · LoyaltyScore / RiskCount / ProtectCount / NetProtection —— 综合评分")
METRICS["n_base_features"] = int(X_base.shape[1])
METRICS["n_eng_features"] = int(X_eng.shape[1])
METRICS["n_new_features"] = int(X_eng.shape[1] - X_base.shape[1])

# 处理派生特征中可能出现的 inf（理论上不存在，但稳健处理）
X_eng = X_eng.replace([np.inf, -np.inf], np.nan)
X_eng = X_eng.fillna(0)

# ======================================================================
# 4. 数据划分
# ======================================================================
log("=" * 70)
log("4. 数据划分（分层抽样，80/20）")
log("=" * 70)
X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
    X_eng, y, test_size=0.2, random_state=RANDOM, stratify=y)
log(f"训练集：{len(X_tr_raw)} 行；测试集：{len(X_te_raw)} 行")
log(f"训练集流失率 {y_tr.mean()*100:.2f}% / 测试集流失率 {y_te.mean()*100:.2f}%")
METRICS["n_train"] = int(len(X_tr_raw))
METRICS["n_test"] = int(len(X_te_raw))

# 区分「原始特征集」与「完整特征集」，用于对比特征工程收益
base_cols = list(X_base.columns)
X_tr_base, X_te_base = X_tr_raw[base_cols], X_te_raw[base_cols]

# ======================================================================
# 5. 建模与调参
# ======================================================================
log("=" * 70)
log("5. 建模与调参（SMOTE 处理类别不平衡 + 网格搜索）")
log("=" * 70)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM)


def build_model(name):
    """返回 (imblearn Pipeline, 参数网格)。Pipeline = 标准化 + SMOTE + 模型。"""
    if name == "逻辑回归":
        est = LogisticRegression(max_iter=3000, random_state=RANDOM)
        grid = {"model__C": [0.01, 0.1, 1, 10]}
    elif name == "随机森林":
        est = RandomForestClassifier(random_state=RANDOM, n_jobs=-1)
        grid = {"model__n_estimators": [200, 400],
                "model__max_depth": [None, 10]}
    elif name == "梯度提升":
        est = GradientBoostingClassifier(random_state=RANDOM)
        grid = {"model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 5]}
    else:
        raise ValueError(name)
    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM)),
        ("model", est),
    ])
    return pipe, grid


# 基线：多数类预测器（用于展示“什么都不学”的水平）
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_tr_raw, y_tr)
dummy_auc = roc_auc_score(y_te, dummy.predict_proba(X_te_raw)[:, 1])
log(f"基线（多数类预测器）测试 AUC：{dummy_auc:.3f}")
METRICS["baseline_auc"] = float(dummy_auc)

models = ["逻辑回归", "随机森林", "梯度提升"]
results = {}   # name -> {best_estimator, best_params, cv_auc, test metrics...}

for mname in models:
    log("-" * 70)
    log(f"训练模型：{mname}")
    pipe, grid = build_model(mname)
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    gs.fit(X_tr_raw, y_tr)
    log(f"  最优参数：{gs.best_params_}")
    log(f"  交叉验证 AUC：{gs.best_score_:.4f}（±{gs.cv_results_['std_test_score'][gs.best_index_]:.4f}）")
    results[mname] = {
        "best_estimator": gs.best_estimator_,
        "best_params": {k.replace("model__", ""): (int(v) if isinstance(v, (int, np.integer)) else v)
                        for k, v in gs.best_params_.items()},
        "cv_auc": float(gs.best_score_),
        "cv_std": float(gs.cv_results_["std_test_score"][gs.best_index_]),
    }

# ======================================================================
# 6. 测试集评估
# ======================================================================
log("=" * 70)
log("6. 测试集评估")
log("=" * 70)
for mname, r in results.items():
    est = r["best_estimator"]
    proba = est.predict_proba(X_te_raw)[:, 1]
    pred = est.predict(X_te_raw)
    r["auc"] = float(roc_auc_score(y_te, proba))
    r["pr_auc"] = float(average_precision_score(y_te, proba))
    r["precision"] = float(precision_score(y_te, pred))
    r["recall"] = float(recall_score(y_te, pred))
    r["f1"] = float(f1_score(y_te, pred))
    r["proba"] = proba
    log(f"{mname}: AUC={r['auc']:.4f}  PR-AUC={r['pr_auc']:.4f}  "
        f"Precision={r['precision']:.4f}  Recall={r['recall']:.4f}  F1={r['f1']:.4f}")
    METRICS[f"{mname}_auc"] = r["auc"]
    METRICS[f"{mname}_pr_auc"] = r["pr_auc"]
    METRICS[f"{mname}_f1"] = r["f1"]
    METRICS[f"{mname}_recall"] = r["recall"]
    METRICS[f"{mname}_precision"] = r["precision"]

# 选出最优模型（按 AUC）
best_name = max(results, key=lambda k: results[k]["auc"])
best = results[best_name]
log(f"最优模型：{best_name}（测试 AUC={best['auc']:.4f}）")
METRICS["best_model"] = best_name

# ---- 6.1 ROC 曲线 ----
fig, ax = plt.subplots(figsize=(7, 5.2))
colors = {"逻辑回归": BLUE, "随机森林": GREEN, "梯度提升": TEAL}
for mname, r in results.items():
    fpr, tpr, _ = roc_curve(y_te, r["proba"])
    ax.plot(fpr, tpr, lw=2.4, color=colors[mname],
            label=f"{mname} (AUC={r['auc']:.3f})")
ax.plot([0, 1], [0, 1], ls="--", color=GRAY, lw=1.2, label="随机猜测")
ax.set_xlabel("假正率 FPR")
ax.set_ylabel("真正率 TPR (Recall)")
ax.set_title("ROC 曲线对比", fontsize=14, fontweight="bold")
ax.legend(loc="lower right")
savefig(fig, "05_roc_curves.png")

# ---- 6.2 PR 曲线 ----
fig, ax = plt.subplots(figsize=(7, 5.2))
for mname, r in results.items():
    prec, rec, _ = precision_recall_curve(y_te, r["proba"])
    ax.plot(rec, prec, lw=2.4, color=colors[mname],
            label=f"{mname} (PR-AUC={r['pr_auc']:.3f})")
ax.set_xlabel("召回率 Recall")
ax.set_ylabel("精确率 Precision")
ax.set_title("精确率-召回率曲线", fontsize=14, fontweight="bold")
ax.legend(loc="lower left")
savefig(fig, "06_pr_curves.png")

# ---- 6.3 最优模型混淆矩阵 ----
best_pred = best["best_estimator"].predict(X_te_raw)
cm = confusion_matrix(y_te, best_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["预测未流失", "预测流失"],
            yticklabels=["实际未流失", "实际流失"],
            annot_kws={"size": 16})
ax.set_title(f"最优模型（{best_name}）混淆矩阵", fontsize=13, fontweight="bold")
ax.set_xlabel("预测")
ax.set_ylabel("实际")
savefig(fig, "07_confusion_matrix.png")
METRICS["confusion_matrix"] = cm.tolist()

# ---- 6.4 分类报告 ----
log("最优模型分类报告：")
log(classification_report(y_te, best_pred, target_names=["未流失", "流失"],
                          digits=4))

# ======================================================================
# 7. 特征重要性 / 模型解释
# ======================================================================
log("=" * 70)
log("7. 特征重要性与解释")
log("=" * 70)
# 7.1 树模型自带的特征重要性（用随机森林）
rf = results["随机森林"]["best_estimator"].named_steps["model"]
imp = pd.Series(rf.feature_importances_, index=X_eng.columns)
imp = imp.sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(7.5, 6))
ax.barh(imp.index[::-1], imp.values[::-1], color=TEAL)
ax.set_xlabel("特征重要性")
ax.set_title("随机森林特征重要性（Top 15）", fontsize=13, fontweight="bold")
savefig(fig, "07_feature_importance.png")
METRICS["top_features"] = list(imp.sort_values(ascending=False).index[:10])

# 7.2 排列重要性（模型无关，用最优模型）
log("计算排列重要性（Permutation Importance）……")
perm = permutation_importance(
    best["best_estimator"], X_te_raw, y_te, scoring="roc_auc",
    n_repeats=10, random_state=RANDOM, n_jobs=-1)
pimp = pd.Series(perm.importances_mean, index=X_eng.columns)
pimp = pimp.sort_values(ascending=False).head(15)
log("排列重要性 Top 10：" + "、".join(pimp.sort_values(ascending=False).index[:10]))
METRICS["top_permutation_features"] = list(pimp.sort_values(ascending=False).index[:10])
fig, ax = plt.subplots(figsize=(7.5, 6))
ax.barh(pimp.index[::-1], pimp.values[::-1], color=AMBER)
ax.set_xlabel("AUC 平均下降量")
ax.set_title("排列重要性（Top 15，模型无关）", fontsize=13, fontweight="bold")
savefig(fig, "09_permutation_importance.png")

# ======================================================================
# 8. 特征工程收益对比（原始特征 vs 原始+工程特征）
# ======================================================================
log("=" * 70)
log("8. 特征工程收益对比")
log("=" * 70)
for mname in models:
    pipe, grid = build_model(mname)
    # 用完整特征集
    gs_full = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    gs_full.fit(X_tr_raw, y_tr)
    auc_full = roc_auc_score(y_te, gs_full.predict_proba(X_te_raw)[:, 1])
    # 用原始特征集
    pipe2, grid2 = build_model(mname)
    gs_base = GridSearchCV(pipe2, grid2, cv=cv, scoring="roc_auc", n_jobs=-1)
    gs_base.fit(X_tr_base, y_tr)
    auc_base = roc_auc_score(y_te, gs_base.predict_proba(X_te_base)[:, 1])
    log(f"{mname}: 原始特征 AUC={auc_base:.4f} → 完整特征 AUC={auc_full:.4f} "
        f"(提升 {auc_full-auc_base:+.4f})")
    METRICS[f"{mname}_auc_base"] = float(auc_base)
    METRICS[f"{mname}_auc_full"] = float(auc_full)

fig, ax = plt.subplots(figsize=(7.5, 4.6))
x = np.arange(len(models))
w = 0.35
base_aucs = [METRICS[f"{m}_auc_base"] for m in models]
full_aucs = [METRICS[f"{m}_auc_full"] for m in models]
ax.bar(x - w/2, base_aucs, w, label="原始特征（18 个）", color=GRAY)
ax.bar(x + w/2, full_aucs, w, label="原始+工程特征", color=TEAL)
for i in range(len(models)):
    ax.text(i - w/2, base_aucs[i] + 0.003, f"{base_aucs[i]:.3f}", ha="center", fontsize=9)
    ax.text(i + w/2, full_aucs[i] + 0.003, f"{full_aucs[i]:.3f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("测试集 ROC-AUC")
ax.set_ylim(0.5, 1.0)
ax.set_title("特征工程带来的提升（原始 vs 完整特征）", fontsize=13, fontweight="bold")
ax.legend()
savefig(fig, "10_engineered_features_effect.png")

# ======================================================================
# 9. 结论与业务建议
# ======================================================================
log("=" * 70)
log("9. 结论与业务建议")
log("=" * 70)
top10 = METRICS["top_features"]
log("最能预测流失的特征（Top 10）：" + "、".join(top10))
log("业务建议：")
log("  · 对「月费高 + 光纤 + 电子支票 + 电子账单 + 无长期合同」的客户重点干预")
log("  · 引导月付客户转长期合同（Contract 是头号保护因子）")
log("  · 对高价值但高风险的客户提供优惠/绑定服务，降低流失")

# 写入指标文件
with open(os.path.join(BASE, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(METRICS, f, ensure_ascii=False, indent=2)

log("=" * 70)
log("完成！图表已保存至 figures/，指标已写入 metrics.json")
log("=" * 70)
