"""
================================================================================
主入口 — CSMAR 真实数据训练
================================================================================
"""
import os, sys, time
import torch, numpy as np, pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_loader"))
from data_interface import HeteroGraphData
from data_loader.csmar_loader import load_csmar_data
from train import RiskWhiteListModel, train, evaluate_model
from config import DEVICE, SEED, OUTPUT_DIR, HIDDEN_DIM

np.random.seed(SEED); torch.manual_seed(SEED)

if __name__ == "__main__":
    print("=" * 60)
    print("  产业链白名单 v4 — CSMAR 真实数据")
    print("=" * 60)

    # ---- 1. 加载 CSMAR 数据 ----
    data = load_csmar_data()

    # ---- 2. 初始化模型（动态适配节点类型和边类型） ----
    print(f"\n[初始化] 模型 v4...")
    in_dims = {}
    for ntype, feat in data.x_dict.items():
        in_dims[ntype] = feat.shape[1]

    model = RiskWhiteListModel(
        in_dims=in_dims,
        edge_types=list(data.edge_index_dict.keys())
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {n_params:,}")
    print(f"  节点类型: {list(in_dims.keys())}")
    print(f"  边类型: {model.edge_names}")
    print(f"  架构: Layer2(SAGE 2层) + Layer3(Γ矩阵) + Layer4(GRU lazy) + 3预测头")

    # ---- 3. 训练 ----
    model = train(model, data)

    # ---- 4. 测试集评估（仅评估有真实标签的上市企业） ----
    print(f"\n[评估] 测试集...")
    test_auc, test_acc, gamma = evaluate_model(model, data, data.test_mask)
    print(f"  测试 AUC: {test_auc:.4f}")
    print(f"  测试 Acc: {test_acc:.4f}")

    # ---- 5. Γ 矩阵输出（可解释性核心） ----
    edge_names = model.edge_names
    print(f"\n[可解释性] Γ 矩阵（关系迁移强度）:")
    header = "              " + "  ".join(f"{e:>10s}" for e in edge_names)
    print(header)
    gamma_np = gamma.cpu().detach().numpy()
    for i, ename_i in enumerate(edge_names):
        vals = "  ".join(f"{gamma_np[i,j]:10.3f}" for j in range(len(edge_names)))
        print(f"  从 {ename_i:>8s}: {vals}")

    # ---- 6. 保存 ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n[保存] → {OUTPUT_DIR}/")
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/model_v4_csmar.pt")
    pd.DataFrame([{"Test_AUC": test_auc, "Test_Acc": test_acc}]
                 ).to_csv(f"{OUTPUT_DIR}/results_csmar.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        gamma_np, index=edge_names, columns=edge_names
    ).to_csv(f"{OUTPUT_DIR}/gamma_matrix_csmar.csv", encoding="utf-8-sig")
    print(f"  [OK] model_v4_csmar.pt  [OK] results_csmar.csv  [OK] gamma_matrix_csmar.csv")

    print(f"\n{'='*60}")
    print(f"  CSMAR 真实数据训练完成。")
    print(f"{'='*60}")
