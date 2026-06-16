"""
================================================================================
入口 — 超图异构双通道模型 v5
================================================================================
用法:
  cd heterogeneous_hypergraph
  python main.py
================================================================================
"""

import sys
import os
import time
import torch
import pandas as pd

# 确保当前目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SEED, DEVICE, OUTPUT_DIR, EDGE_TYPES,
)
from data_loader.csmar_loader import load_csmar_data_v5
from train import HyperHeteroModel, train, evaluate_model

torch.manual_seed(SEED)


def main():
    print("=" * 60)
    print("  超图异构双通道模型 v5 — CSMAR 真实数据")
    print("=" * 60)

    # ── Step 1: 加载数据 ──
    print("\n[Step 1] 加载 CSMAR 数据 (v5 管线)...")
    t0 = time.time()
    data = load_csmar_data_v5()
    print(f"  数据加载完成，总耗时 {time.time() - t0:.1f}s")

    # ── Step 2: 构建模型 ──
    print("\n[Step 2] 构建模型...")
    in_dims = {ntype: data.x_dict[ntype].shape[1] for ntype in data.x_dict}

    # 过滤：只保留数据中实际存在的边类型
    valid_edge_types = [et for et in EDGE_TYPES if et in data.edge_index_dict]
    if valid_edge_types != EDGE_TYPES:
        missing = set(EDGE_TYPES) - set(valid_edge_types)
        print(f"  注意: 以下边类型在数据中不存在，跳过: {missing}")

    model = HyperHeteroModel(in_dims=in_dims, edge_types=valid_edge_types)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params:,}")
    print(f"  节点类型: {list(in_dims.keys())}")
    print(f"  边类型: {[et[1] for et in valid_edge_types]}")
    print(f"  超图视图: {list(data.hyperedges.keys())}")
    print(f"  架构: 超图4视图 + 异构{len(in_dims)}节点{len(valid_edge_types)}边 + FusionGate + Γ + GRU")

    # ── Step 3: 训练 ──
    print(f"\n[Step 3] 训练...")
    model = train(model, data)

    # ── Step 4: 测试集评估 ──
    print(f"\n[Step 4] 测试集评估...")
    test_auc, test_acc, test_prec10, gamma = evaluate_model(model, data, data.test_mask)
    print(f"  测试 AUC: {test_auc:.4f}")
    print(f"  测试 Acc: {test_acc:.4f}")
    print(f"  Precision@10: {test_prec10:.4f}")

    # ── Step 5: Γ 矩阵 ──
    print(f"\n[Step 5] Γ 矩阵（关系迁移强度）:")
    gamma_np = gamma.cpu().numpy()
    edge_names = list(set(et[1] for et in valid_edge_types))
    R = len(edge_names)
    header = "           " + "".join(f"{n:>10s}" for n in edge_names)
    print(header)
    for i, name_i in enumerate(edge_names):
        row = f"  {name_i:>10s} " + "".join(f"{gamma_np[i, j]:10.3f}" for j in range(R))
        print(row)

    # ── Step 6: 保存 ──
    print(f"\n[Step 6] 保存到 {OUTPUT_DIR}/ ...")
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/model_v5.pt")
    pd.DataFrame({"test_auc": [test_auc], "test_acc": [test_acc], "precision_at_10": [test_prec10]}).to_csv(
        f"{OUTPUT_DIR}/results_v5.csv", index=False
    )
    pd.DataFrame(gamma_np, index=edge_names, columns=edge_names).to_csv(
        f"{OUTPUT_DIR}/gamma_matrix_v5.csv"
    )
    print(f"  [OK] model_v5.pt  [OK] results_v5.csv  [OK] gamma_matrix_v5.csv")

    print("\n" + "=" * 60)
    print(f"  训练完成。Test AUC = {test_auc:.4f}, Precision@10 = {test_prec10:.4f}")
    print(f"  对照: v4 GNN Test AUC = 0.7619, v5 XGBoost = 0.7362")
    print("=" * 60)


if __name__ == "__main__":
    main()
