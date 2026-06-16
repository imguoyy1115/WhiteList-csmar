"""
================================================================================
消融实验 A: 去掉 Γ 矩阵（跨关系风险传播）
================================================================================
对照 main.py 的三层架构，将 Γ 矩阵退化为 I（单位矩阵），
测试跨关系风险传播是否对模型有真实贡献。

即: 去掉 Layer 3 (CrossRelationPropagation) 的跨关系迁移功能，
    保留语义注意力融合。

用法:
  cd heterogeneous_hypergraph
  python main_no_gamma.py

对比: main.py (Test AUC 目标 0.7949)
================================================================================
"""

import sys
import os
import time
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
# ── 消融开关：关闭 Γ ──
config.ABLATION_NO_GAMMA = True

from config import (
    SEED, DEVICE, OUTPUT_DIR, EDGE_TYPES,
)
from data_loader.csmar_loader import load_csmar_data_v5
from train import HyperHeteroModel, train, evaluate_model

torch.manual_seed(SEED)


def main():
    print("=" * 60)
    print("  消融实验: 去掉 Γ 矩阵（Γ = I，无跨关系传播）")
    print("=" * 60)

    # ── Step 1: 加载数据 ──
    print("\n[Step 1] 加载 CSMAR 数据 (v5 管线)...")
    t0 = time.time()
    data = load_csmar_data_v5()
    print(f"  数据加载完成，总耗时 {time.time() - t0:.1f}s")

    # ── Step 2: 构建模型 ──
    print("\n[Step 2] 构建模型（Γ = I，无跨关系传播）...")
    in_dims = {ntype: data.x_dict[ntype].shape[1] for ntype in data.x_dict}

    valid_edge_types = [et for et in EDGE_TYPES if et in data.edge_index_dict]
    if valid_edge_types != EDGE_TYPES:
        missing = set(EDGE_TYPES) - set(valid_edge_types)
        print(f"  注意: 以下边类型在数据中不存在，跳过: {missing}")

    model = HyperHeteroModel(in_dims=in_dims, edge_types=valid_edge_types)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params:,}")
    print(f"  架构: 超图4视图 + 异构{len(in_dims)}节点{len(valid_edge_types)}边 + FusionGate + Γ=I(消融) + GRU")

    # ── Step 3: 训练 ──
    print(f"\n[Step 3] 训练...")
    model = train(model, data)

    # ── Step 4: 测试集评估 ──
    print(f"\n[Step 4] 测试集评估...")
    test_auc, test_acc, test_prec10, gamma = evaluate_model(model, data, data.test_mask)
    print(f"  测试 AUC: {test_auc:.4f}")
    print(f"  测试 Acc: {test_acc:.4f}")
    print(f"  Precision@10: {test_prec10:.4f}")

    # ── Step 5: Γ 矩阵（消融模式下为 I） ──
    print(f"\n[Step 5] Γ 矩阵（消融模式，应为单位矩阵）:")
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
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/model_no_gamma.pt")
    pd.DataFrame({
        "experiment": ["no_gamma"],
        "test_auc": [test_auc],
        "test_acc": [test_acc],
        "precision_at_10": [test_prec10]
    }).to_csv(f"{OUTPUT_DIR}/results_no_gamma.csv", index=False)
    print(f"  [OK] model_no_gamma.pt  [OK] results_no_gamma.csv")

    print("\n" + "=" * 60)
    print(f"  消融完成。Test AUC (无Γ) = {test_auc:.4f}, Precision@10 = {test_prec10:.4f}")
    print(f"  等待对照 main.py (含Γ) 的结果做差: Δ = AUC_main - AUC_nogamma")
    print("=" * 60)


if __name__ == "__main__":
    main()
