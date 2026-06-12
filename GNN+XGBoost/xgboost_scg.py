"""
================================================================================
XGBoost Baseline — 用 SCG 供应链数据集跑产品分类
================================================================================

数据集：SCG (Supply Chain Graph) — 孟加拉国快消品供应链
来源：GNN in Supply Chain Analytics Benchmarks (arXiv:2411.08550)

数据说明：
  - 41 个产品节点（不同包装规格/口味的薯片等快消品）
  - 每个产品属于一个 Product Group：S / P / A / M / E（5 类）
  - 时序数据：2023-01-01 ~ 2023-08-09（222 天），每天一条记录
    · Sales Order      — 销售订单量（分销商请求量）
    · Production       — 实际生产量
    · Delivery to Dist — 已交付量
    · Factory Issue    — 工厂出货量

XGBoost 任务：
  输入：每个产品的时序统计特征（均值、方差、趋势等，~40 维）
  输出：产品属于哪个 Product Group（5 分类）

赛题对标：
  SCG 上 XGBoost 做产品分类
  约等于
  赛题数据上 XGBoost 做白名单分类
  → 都是用表格特征做分类，看不到图结构

后期切换：
  赛题数据到手 → 注释掉 load_scg_data()
  → 取消注释 load_real_data() → 重跑即可

环境依赖：
  pip install xgboost scikit-learn pandas numpy

================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings
import time
import os

warnings.filterwarnings("ignore")

# ============================================================================
# 全局配置
# ============================================================================
SEED = 42
np.random.seed(SEED)

# SCG 数据集路径（你本地已有的）
SCG_DATA_DIR = (
    r"D:/Users/imguoyyy/PycharmProjects/WhiteList/SCG_Dataset"
)

OUTPUT_DIR = "xgboost_outputs_scg"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# Step 1: 加载 SCG 供应链数据集
# ============================================================================
def load_scg_data(data_dir=SCG_DATA_DIR):
    """
    ==========================================================================
    从 SCG 原始 CSV 文件中提取表格特征，供 XGBoost 使用。

    做法：
      对每个产品，从 222 天的时序数据中计算统计特征——
      均值、标准差、最小值、最大值、最近 30 天均值、线性趋势斜率。
      4 种时序 × 6 个统计量 = 24 维时序特征。

      再加图结构衍生特征（不从图卷积来，只是从边表统计）——
      同组产品数、同工厂产品数、同仓库产品数 = 3 维。

      总共 ~27 维特征（真实赛题数据会更多，约 100 维）。

    返回：
      X:        (41, n_features) 特征矩阵
      y:        (41,) 标签，Product Group 编码为 0~4
      y_labels: list，标签的原始名称 ['S', 'P', 'A', 'M', 'E']
      feature_names: list，特征名列表
    ==========================================================================
    """
    base = f"{data_dir}/homogenoeus"

    # ---- 1.1 读取产品列表和其 Product Group 标签 ----
    node_types = pd.read_csv(f"{base}/Nodes/Node Types (Product Group and Subgroup).csv")
    # node_types: Node, Group, Sub-Group
    # 例如 SOS008L02P, S, SOS

    node_index = pd.read_csv(f"{base}/Nodes/NodesIndex.csv")
    # node_index: Node, NodeIndex (0~40)

    # 合并得到每个 NodeIndex 的 Group
    node_info = node_index.merge(node_types, on="Node", how="left")
    # 按 NodeIndex 排序
    node_info = node_info.sort_values("NodeIndex").reset_index(drop=True)

    product_names = node_info["Node"].values       # 产品代码，如 SOS008L02P
    group_labels_raw = node_info["Group"].values   # 原始标签
    sub_group = node_info["Sub-Group"].values

    # 标签编码：S→0, P→1, A→2, M→3, E→4
    le = LabelEncoder()
    y = le.fit_transform(group_labels_raw)
    y_labels = list(le.classes_)
    n_samples = len(y)

    print(f"[Step 1] 加载 SCG 供应链数据集...")
    print(f"  产品数: {n_samples}")
    print(f"  产品组: {y_labels}  (共 {len(y_labels)} 类)")
    for i, lbl in enumerate(y_labels):
        print(f"    {lbl}: {(y == i).sum()} 个产品")

    # ---- 1.2 从时序数据中提取静态特征 ----
    # 4 种时序：Sales Order, Production, Delivery, Factory Issue

    temporal_dir = f"{base}/Temporal Data/Unit"
    temporal_files = {
        "SalesOrder":   f"{temporal_dir}/Sales Order.csv",
        "Production":   f"{temporal_dir}/Production .csv",
        "Delivery":     f"{temporal_dir}/Delivery To distributor.csv",
        "FactoryIssue": f"{temporal_dir}/Factory Issue.csv",
    }

    n_sources = len(temporal_files)  # 4
    n_stats = 6
    n_features_temporal = n_sources * n_stats  # 24
    X = np.zeros((n_samples, n_features_temporal), dtype=np.float32)

    # 生成特征名（每对 数据源×统计量 一个名字，跟产品的列无关）
    feature_names = []
    stat_names = ["mean", "std", "min", "max", "recent30", "trend"]
    for data_name in temporal_files.keys():
        for sn in stat_names:
            feature_names.append(f"{data_name}_{sn}")

    # 对每个产品、每种时序数据，计算 6 个统计量，填入 X
    for src_idx, (data_name, filepath) in enumerate(temporal_files.items()):
        df_temporal = pd.read_csv(filepath)
        value_cols = list(df_temporal.columns[1:])  # 去掉 Date 列

        for prod_idx, product_name in enumerate(product_names):
            col_start = src_idx * n_stats

            if product_name in value_cols:
                series = pd.to_numeric(
                    df_temporal[product_name], errors="coerce"
                ).dropna()
                if len(series) > 0:
                    stats = [
                        series.mean(),
                        series.std() if len(series) > 1 else 0.0,
                        series.min(),
                        series.max(),
                        series.iloc[-30:].mean() if len(series) >= 30 else series.mean(),
                        _trend_slope(series),
                    ]
                else:
                    stats = [0.0] * n_stats
            else:
                stats = [0.0] * n_stats

            X[prod_idx, col_start : col_start + n_stats] = stats

    # ---- 1.3 图结构衍生特征（从边表统计，不需 GNN）----
    # 跟赛题一样，XGBoost 可以拿到一些"静态图统计"作为特征
    # 同组产品数、同子组产品数、同工厂产品数、同仓库产品数
    graph_feat = _extract_graph_features(base, product_names, n_samples)
    X = np.hstack([X, graph_feat.astype(np.float32)])

    graph_feature_names = [
        "same_group_count",      # 同 Product Group 的产品数
        "same_subgroup_count",   # 同 Sub-Group 的产品数
        "same_plant_count",      # 同工厂的产品数
        "same_storage_count",    # 同仓库的产品数
    ]
    feature_names.extend(graph_feature_names)

    print(f"  特征维度: {X.shape[1]} (时序 {n_features} + 图结构 {len(graph_feature_names)})")
    print(f"  标签分布: {dict(zip(y_labels, np.bincount(y)))}")

    return X, y, y_labels, feature_names, product_names, node_info


def _trend_slope(series):
    """计算时序的线性趋势斜率（正=增长，负=下降）"""
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series))
    slope = np.polyfit(x, series.values, 1)[0]
    return float(slope)


def _extract_graph_features(base_dir, product_names, n_samples):
    """从边表中提取图结构统计特征（XGBoost 拿不到图消息传递，但能拿到这些统计量）"""
    # 同工厂产品数
    plant_edges = pd.read_csv(f"{base_dir}/Edges/EdgesIndex/Edges (Plant).csv")
    plant_degree = plant_edges.groupby("node1").size().to_dict()
    # 同产品组产品数
    group_edges = pd.read_csv(f"{base_dir}/Edges/EdgesIndex/Edges (Product Group).csv")
    group_degree = group_edges.groupby("node1").size().to_dict()
    # 同子组产品数
    sub_edges = pd.read_csv(f"{base_dir}/Edges/EdgesIndex/Edges (Product Sub-Group).csv")
    sub_degree = sub_edges.groupby("node1").size().to_dict()
    # 同仓库产品数
    storage_edges = pd.read_csv(f"{base_dir}/Edges/EdgesIndex/Edges (Storage Location).csv")
    storage_degree = storage_edges.groupby("node1").size().to_dict()

    graph_feat = np.zeros((n_samples, 4))
    for i in range(n_samples):
        graph_feat[i, 0] = group_degree.get(i, 0)
        graph_feat[i, 1] = sub_degree.get(i, 0)
        graph_feat[i, 2] = plant_degree.get(i, 0)
        graph_feat[i, 3] = storage_degree.get(i, 0)

    return graph_feat


# ============================================================================
# Step 2: 数据切分
# ============================================================================
def split_data(X, y, test_size=0.25, val_size=0.15):
    """分层切分（保证每类在训练/验证/测试集中比例一致）"""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=SEED, stratify=y_temp
    )
    print(f"\n[Step 2] 数据切分:")
    print(f"  训练: {len(X_train)} | 验证: {len(X_val)} | 测试: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================================
# Step 3: XGBoost 训练
# ============================================================================
def train_xgboost(X_train, y_train, X_val, y_val, num_classes):
    """
    多分类 XGBoost 训练。
    如果你的赛题是二分类，把 objective 改成 'binary:logistic' 即可。
    """
    print(f"\n[Step 3] XGBoost 训练（{num_classes} 分类）...")

    params = {
        "n_estimators": 200,
        "max_depth": 4,                     # 样本少(41)，深度不能太大
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "multi:softprob",      # 多分类
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "random_state": SEED,
        "early_stopping_rounds": 30,
        "verbosity": 0,
    }

    start = time.time()
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    elapsed = time.time() - start

    print(f"  训练完成，耗时 {elapsed:.1f}s")
    print(f"  最佳迭代: {model.best_iteration}")
    return model


# ============================================================================
# Step 4: 评估
# ============================================================================
def evaluate(model, X_test, y_test, y_labels, feature_names):
    """多分类评估 + 每类的 AUC（One-vs-Rest）"""
    print(f"\n[Step 4] 测试集评估...")
    print("=" * 60)

    y_pred_proba = model.predict_proba(X_test)  # (N, num_classes)
    y_pred = model.predict(X_test)              # (N,)

    # ---- 4.1 整体指标 ----
    acc = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"  Accuracy:        {acc:.4f}")
    print(f"  Precision(macro):{precision_macro:.4f}")
    print(f"  Recall(macro):   {recall_macro:.4f}")
    print(f"  F1(macro):       {f1_macro:.4f}")

    # ---- 4.2 每类 AUC（OvR）----
    print(f"\n  --- 每类 AUC (One-vs-Rest) ---")
    auc_per_class = {}
    for i, label in enumerate(y_labels):
        y_true_bin = (y_test == i).astype(int)
        y_score = y_pred_proba[:, i]
        try:
            auc = roc_auc_score(y_true_bin, y_score)
            auc_per_class[label] = auc
            print(f"  {label}: AUC={auc:.4f}")
        except ValueError:
            print(f"  {label}: AUC=N/A (该类在测试集中只有一种标签)")

    # ---- 4.3 混淆矩阵 ----
    print(f"\n  --- 混淆矩阵 ---")
    # labels= 参数保证矩阵维度固定，即使测试集中缺少某个类也不会报错
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(y_labels))))
    header = "         " + "".join(f"  预测{l:>4s}" for l in y_labels)
    print(header)
    for i, label in enumerate(y_labels):
        row = f"  实际{label:>4s} " + "".join(f"{cm[i,j]:>10d}" for j in range(len(y_labels)))
        print(row)

    # ---- 4.4 分类报告 ----
    print(f"\n  --- 分类报告 ---")
    print(classification_report(y_test, y_pred, target_names=y_labels,
                                 labels=list(range(len(y_labels))), zero_division=0))

    # ---- 4.5 特征重要性 ----
    print(f"\n  --- 特征重要性 Top-10 (gain) ---")
    importance = model.get_booster().get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for rank, (fname, score) in enumerate(sorted_imp, 1):
        idx = int(fname.replace("f", ""))
        real_name = feature_names[idx] if idx < len(feature_names) else fname
        print(f"  {rank:>2}. {real_name:<30s} {score:>10.2f}")

    results = {
        "Accuracy": acc,
        "Precision_macro": precision_macro,
        "Recall_macro": recall_macro,
        "F1_macro": f1_macro,
        **{f"AUC_{l}": auc_per_class.get(l, np.nan) for l in y_labels},
    }
    return results, y_pred_proba, y_pred


# ============================================================================
# Step 5: 保存结果
# ============================================================================
def save_results(results, model, feature_names):
    """保存指标、模型、特征重要性"""
    print(f"\n[Step 5] 保存结果到 {OUTPUT_DIR}/ ...")

    pd.DataFrame([results]).to_csv(
        f"{OUTPUT_DIR}/metrics.csv", index=False, encoding="utf-8-sig"
    )
    model.save_model(f"{OUTPUT_DIR}/xgboost_model.json")

    imp = model.get_booster().get_score(importance_type="gain")
    imp_df = pd.DataFrame([
        {"feature": feature_names[int(k.replace("f", ""))], "gain": v}
        for k, v in imp.items()
    ]).sort_values("gain", ascending=False)
    imp_df.to_csv(
        f"{OUTPUT_DIR}/feature_importance.csv", index=False, encoding="utf-8-sig"
    )
    print(f"  ✓ metrics.csv  ✓ xgboost_model.json  ✓ feature_importance.csv")


# ============================================================================
# 主流程
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  XGBoost Baseline — SCG 供应链产品分类")
    print("=" * 60)

    # ---- Step 1: 加载数据 ----
    X, y, y_labels, feature_names, product_names, node_info = load_scg_data()

    # 【赛题数据到手后】：注释掉上面一行，取消注释下面这行
    # X, y, y_labels, feature_names, _, _ = load_real_data("你的赛题数据路径")
    # 注意：赛题数据是二分类(y∈{0,1})，下面 train_xgboost 的 objective 要改

    # ---- Step 2: 切分 ----
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # ---- Step 3: 训练 ----
    num_classes = len(y_labels)
    model = train_xgboost(X_train, y_train, X_val, y_val, num_classes)

    # ---- Step 4: 评估 ----
    results, y_pred_proba, y_pred = evaluate(
        model, X_test, y_test, y_labels, feature_names
    )

    # ---- Step 5: 保存 ----
    save_results(results, model, feature_names)

    # ---- 总结 ----
    print("\n" + "=" * 60)
    print("  赛题对标说明")
    print("=" * 60)
    print(f"  当前任务: SCG 产品 5 分类 (XGBoost)")
    print(f"  赛题任务:   产业链企业白名单 2 分类 (XGBoost)")
    print(f"  共同点:     都用表格特征做分类，都不使用图消息传递")
    print(f"  差异:       多分类→二分类（改 objective 参数即可）")
    print(f"              41样本→10万样本（数据量大很多）")
    print(f"              时序特征→企业特征（特征含义不同，维度类似）")
    print(f"")
    print(f"  当前 XGBoost 结果仅供参考流程验证，")
    print(f"  赛题数据到手后替换 Step 1 即可重新训练。")
    print("=" * 60)
