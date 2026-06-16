"""
================================================================================
XGBoost Baseline — CSMAR 产业链白名单二分类
================================================================================

对标 model_v4 GNN 的纯表格 baseline：
  - 仅使用原始企业特征 X（25 维），不做任何图对齐/邻域聚合/时序展开
  - 使用完全相同的 train/val/test 切分
  - 不做图消息传递，纯 XGBoost 表格模型

对比逻辑：
  GNN AUC - XGBoost AUC = 图消息传递带来的真实增益
  （XGBoost 拿不到图结构，GNN 如果赢了才是图的价值）

特征说明：
  25 维全部来自每家企业自己的 CSMAR 数据（财务指标 + SCF + 诉讼），
  不包含 struct_hint（度数等图统计量）、X_struct（邻居特征聚合）、
  x_seq（时序快照），确保 baseline 不含任何图信息泄露。

数据来源：
  复用 model_v4/data_loader/csmar_loader.py 的完整数据管线

用法：
  python xgboost_baseline.py
================================================================================
"""

import numpy as np
import pandas as pd
import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

# ── 导入 model_v4 数据管线 ──
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_v4"))
from data_loader.csmar_loader import load_csmar_data

from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
import xgboost as xgb


def precision_at_k(y_true, y_score, k: int = 10):
    """Precision@K: 模型打分最高的 K 个样本中正类的比例"""
    if len(y_score) == 0:
        return 0.0
    k = min(k, len(y_score))
    top_k_idx = np.argsort(y_score)[-k:]
    return float(y_true[top_k_idx].sum()) / k

# ============================================================================
# 全局配置
# ============================================================================
SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xgboost_outputs_csmar")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# XGBoost 超参数（针对 ~4000 样本，162 维特征，二分类）
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 5,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": SEED,
    "early_stopping_rounds": 50,
    "verbosity": 0,
}


# ============================================================================
# Step 1: 加载 CSMAR 数据，构建表格特征矩阵
# ============================================================================
def build_tabular_features():
    """
    ==========================================================================
    调用 model_v4 的数据管线，拿到跟 GNN 完全相同的原始数据，
    然后展平成 (N, F) 表格供 XGBoost 使用。

    特征构成（162 维）：
      X_raw:        (N, 25)  原始企业特征（财务 + 交易信用 + SCF + 诉讼）
      X_struct:     (N, 25)  邻域聚合特征（邻居特征均值，跟 GNN gate 输入相同）
      struct_hint:  (N, 8)   图结构统计量（度、是否上市、特征覆盖率等）
      x_seq_flat:   (N, 104) 季度序列展平（4 季 × 26 维）
    ==========================================================================
    """
    print("=" * 60)
    print("  XGBoost Baseline — CSMAR 产业链白名单二分类")
    print("=" * 60)

    print("\n[Step 1] 加载 CSMAR 数据（复用 GNN 数据管线）...")
    t0 = time.time()
    data = load_csmar_data()
    print(f"  加载完成，耗时 {time.time() - t0:.1f}s")

    # ── 只用最原始的企业特征，不做任何图对齐/邻域聚合/时序展开 ──
    #     X_raw 里的 25 维全部来自每家企业自己的 CSMAR 数据（财务 + SCF + 诉讼）
    #     不包含 X_struct（邻居聚合）、struct_hint（度数）、x_seq（时序）
    #     这样 GNN 如果赢了，才是图消息传递的真实贡献
    X = data.x_dict["enterprise"].numpy().astype(np.float32)            # (N, 25)
    n_features = X.shape[1]

    # ── 标签 & Mask ──
    y = data.y_white.numpy().astype(int)               # (N,) 0/1
    train_mask = data.train_mask.numpy().astype(bool)
    val_mask = data.val_mask.numpy().astype(bool)
    test_mask = data.test_mask.numpy().astype(bool)

    # ── 特征名 ──
    feature_names = [
        "trade_credit_power",        # 0  话语权
        "trade_credit_days",         # 1  信用天数
        "scf_credit_flag",           # 2  SCF授信
        "scf_ar_ratio",              # 3  应收账款占比
        "scf_ap_ratio",              # 4  预付账款占比
        "scf_ar_avg_amount",         # 5  应收账款均值
        "scf_ap_avg_amount",         # 6  预付账款均值
        "scf_overview_flag",         # 7  SCF总体
        "scf_stats_flag",            # 8  SCF统计
        "solvency_1",                # 9  偿债能力
        "solvency_2",                # 10
        "profit_1",                  # 11 盈利能力
        "profit_2",                  # 12
        "operation_1",               # 13 经营能力
        "operation_2",               # 14
        "growth_1",                  # 15 发展能力
        "growth_2",                  # 16
        "cashflow_1",                # 17 现金流
        "cashflow_2",                # 18
        "risklevel_1",               # 19 风险水平
        "risklevel_2",               # 20
        "revenue_growth",            # 21 营收增长
        "asset_turnover",            # 22 资产周转
        "lawsuit_total_amount",      # 23 诉讼累计金额 (log1p)
        "lawsuit_weighted_severity", # 24 诉讼加权严重分 (log1p)
    ]

    print(f"  特征维度: {n_features} (纯原始企业特征，无图结构信息)")
    print(f"  训练集: {train_mask.sum()} | 验证集: {val_mask.sum()} | 测试集: {test_mask.sum()}")
    print(f"  训练集正样本比例: {y[train_mask].mean():.3f}")
    print(f"  测试集正样本比例: {y[test_mask].mean():.3f}")

    return X, y, train_mask, val_mask, test_mask, feature_names


# ============================================================================
# Step 2: XGBoost 训练
# ============================================================================
def train_xgboost(X_train, y_train, X_val, y_val):
    """
    二分类 XGBoost，使用验证集做早停。
    自动计算 scale_pos_weight 处理类别不平衡。
    """
    print(f"\n[Step 2] XGBoost 训练...")

    # 类别不平衡：自动计算权重
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    params = XGB_PARAMS.copy()
    params["scale_pos_weight"] = scale_pos_weight
    # xgboost 3.x: early_stopping_rounds 在构造函数里，不在 fit() 里
    params.pop("verbosity", None)
    early_stop = params.pop("early_stopping_rounds", 50)

    print(f"  训练样本: {len(X_train)}, 正样本比例: {y_train.mean():.3f}")
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")
    print(f"  max_depth: {XGB_PARAMS['max_depth']}, lr: {XGB_PARAMS['learning_rate']}")
    print(f"  n_estimators: {XGB_PARAMS['n_estimators']}, early_stopping: {early_stop}")

    t0 = time.time()
    model = xgb.XGBClassifier(
        early_stopping_rounds=early_stop,
        verbosity=0,
        **params,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    elapsed = time.time() - t0

    best_iter = getattr(model, "best_iteration", None) or XGB_PARAMS["n_estimators"]
    print(f"  训练完成，耗时 {elapsed:.1f}s")
    print(f"  最佳迭代: {best_iter}")
    print(f"  最佳验证 AUC: {model.best_score:.4f}")

    return model


# ============================================================================
# Step 3: 评估
# ============================================================================
def evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, feature_names):
    """二分类评估：AUC + Accuracy + F1 + 特征重要性"""
    print(f"\n[Step 3] 评估...")
    print("=" * 60)

    y_pred_proba = model.predict_proba(X_test)[:, 1]  # 正类概率
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # ── 测试集指标 ──
    test_auc = roc_auc_score(y_test, y_pred_proba)
    test_acc = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, zero_division=0)
    test_recall = recall_score(y_test, y_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    test_prec10 = precision_at_k(y_test, y_pred_proba, k=10)

    print(f"  ── 测试集 ──")
    print(f"  AUC:          {test_auc:.4f}")
    print(f"  Accuracy:     {test_acc:.4f}")
    print(f"  Precision:    {test_precision:.4f}")
    print(f"  Recall:       {test_recall:.4f}")
    print(f"  F1:           {test_f1:.4f}")
    print(f"  Precision@10: {test_prec10:.4f}")

    # ── 训练集 & 验证集 AUC（辅助诊断过拟合） ──
    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]
    train_auc = roc_auc_score(y_train, train_proba)
    val_auc = roc_auc_score(y_val, val_proba)
    print(f"\n  ── 辅助 ──")
    print(f"  训练 AUC:  {train_auc:.4f}")
    print(f"  验证 AUC:  {val_auc:.4f}")
    print(f"  测试 AUC:  {test_auc:.4f}")
    print(f"  Train-Test Gap: {train_auc - test_auc:.4f}")

    # ── 混淆矩阵 ──
    print(f"\n  ── 混淆矩阵 ──")
    cm = confusion_matrix(y_test, y_pred)
    print(f"             预测负类  预测正类")
    print(f"  实际负类  {cm[0,0]:>10d}  {cm[0,1]:>10d}")
    print(f"  实际正类  {cm[1,0]:>10d}  {cm[1,1]:>10d}")

    # ── 特征重要性 Top-20 ──
    print(f"\n  ── 特征重要性 Top-20 (gain) ──")
    importance = model.get_booster().get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
    for rank, (fname, score) in enumerate(sorted_imp, 1):
        idx = int(fname.replace("f", ""))
        real_name = feature_names[idx] if idx < len(feature_names) else fname
        print(f"  {rank:>2}. {real_name:<40s} {score:>10.2f}")

    results = {
        "Test_AUC": test_auc,
        "Test_Accuracy": test_acc,
        "Test_Precision": test_precision,
        "Test_Recall": test_recall,
        "Test_F1": test_f1,
        "Precision_at_10": test_prec10,
        "Train_AUC": train_auc,
        "Val_AUC": val_auc,
        "Best_Iteration": model.best_iteration,
        "Train_Pos_Ratio": y_train.mean(),
        "Test_Pos_Ratio": y_test.mean(),
    }
    return results, y_pred_proba, y_pred


# ============================================================================
# Step 4: 保存
# ============================================================================
def save_results(results, model, feature_names, X, y, test_mask, y_pred_proba):
    """保存指标、模型、特征重要性、预测结果"""
    print(f"\n[Step 4] 保存结果到 {OUTPUT_DIR}/ ...")

    # 指标
    pd.DataFrame([results]).to_csv(
        f"{OUTPUT_DIR}/metrics.csv", index=False, encoding="utf-8-sig"
    )

    # 模型
    model.save_model(f"{OUTPUT_DIR}/xgboost_model.json")

    # 特征重要性
    imp = model.get_booster().get_score(importance_type="gain")
    imp_df = pd.DataFrame([
        {"feature": feature_names[int(k.replace("f", ""))], "gain": v}
        for k, v in imp.items()
    ]).sort_values("gain", ascending=False)
    imp_df.to_csv(
        f"{OUTPUT_DIR}/feature_importance.csv", index=False, encoding="utf-8-sig"
    )

    # 测试集预测详情
    test_idx = np.where(test_mask)[0]
    pred_df = pd.DataFrame({
        "global_id": test_idx,
        "y_true": y[test_idx].astype(int),
        "y_proba": y_pred_proba,
        "y_pred": (y_pred_proba >= 0.5).astype(int),
    })
    pred_df.to_csv(
        f"{OUTPUT_DIR}/test_predictions.csv", index=False, encoding="utf-8-sig"
    )

    print(f"  [OK] metrics.csv  [OK] xgboost_model.json")
    print(f"  [OK] feature_importance.csv  [OK] test_predictions.csv")


# ============================================================================
# 主流程
# ============================================================================
if __name__ == "__main__":
    # ── Step 1: 加载数据 ──
    X, y, train_mask, val_mask, test_mask, feature_names = build_tabular_features()

    # ── Step 2: 按 mask 切分 ──
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    # ── Step 3: 训练 ──
    model = train_xgboost(X_train, y_train, X_val, y_val)

    # ── Step 4: 评估 ──
    results, y_pred_proba, y_pred = evaluate(
        model, X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    )

    # ── Step 5: 保存 ──
    save_results(results, model, feature_names, X, y, test_mask, y_pred_proba)

    # ── 对比总结 ──
    print("\n" + "=" * 60)
    print("  GNN vs XGBoost 对比（公平 baseline）")
    print("=" * 60)
    print(f"              Test AUC")
    print(f"  XGBoost     {results['Test_AUC']:.4f}  (纯表格 25维，无图信息)")
    print(f"  GNN v4.3    0.7619  (SAGE 2层 + Γ + GRU TemporalGate)")
    print(f"  ─────────────────────")
    gap = 0.7619 - results['Test_AUC']
    status = "← GNN 学到了图结构的额外信息" if gap > 0 else "← GNN 未超越纯表格 baseline"
    print(f"  Δ (GNN增益)  {gap:+.4f}  {status}")
    print(f"\n  说明:")
    print(f"    XGBoost: 仅 X_raw (25 维原始企业特征)")
    print(f"    GNN:      X_raw + 2-hop SAGEConv 图消息传递 + Γ 跨关系风险传播")
    print(f"    切分:     复用 GNN 的 train/val/test mask（完全对齐）")
    print(f"    含义:     {'GNN 的图结构有正向价值' if gap > 0 else '当前 GNN 训练方式未充分挖掘图信息'}")
    print("=" * 60)
