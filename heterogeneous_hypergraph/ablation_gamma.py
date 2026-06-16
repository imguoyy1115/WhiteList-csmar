"""
================================================================================
Γ 矩阵消融实验
================================================================================
测试 Cross-Relation Risk Propagation 是否对模型有真实贡献。

实验设计：
  实验组 A:  Γ 可学习（正常模式）→ 跨关系风险传播生效
  对照组 B:  Γ = I（消融模式）   → 风险不跨关系迁移，仅语义注意力融合

如果 A 的 AUC 显著高于 B → Γ 矩阵有真实贡献。
如果 A ≈ B                 → Γ 矩阵是冗余设计，跨关系传播对当前数据无增益。

用法:
  cd heterogeneous_hypergraph
  python ablation_gamma.py
================================================================================
"""

import sys
import os
import time
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_loader.csmar_loader import load_csmar_data_v5
from train import HyperHeteroModel, train, evaluate_model


def run_experiment(name: str, no_gamma: bool, data, edge_types):
    """运行单次实验"""
    print("\n" + "=" * 60)
    print(f"  {name}")
    print(f"  Γ 矩阵: {'关闭 (Γ=I, 无跨关系传播)' if no_gamma else '开启 (可学习)'}")
    print("=" * 60)

    # 设置消融标志（运行时读取，影响模型构造函数）
    config.ABLATION_NO_GAMMA = no_gamma

    # 构建模型
    in_dims = {ntype: data.x_dict[ntype].shape[1] for ntype in data.x_dict}
    model = HyperHeteroModel(in_dims=in_dims, edge_types=edge_types)

    # 训练
    t0 = time.time()
    model = train(model, data)
    train_time = time.time() - t0

    # 测试集评估
    model.eval()
    test_auc, test_acc, gamma_final = evaluate_model(model, data, data.test_mask)
    val_auc, val_acc, _ = evaluate_model(model, data, data.val_mask)

    # Γ 矩阵统计
    if not no_gamma and gamma_final is not None:
        diag_mean = gamma_final.diag().mean().item()
        off_diag_mean = (gamma_final.sum() - gamma_final.diag().sum()).item() / \
                        (gamma_final.numel() - gamma_final.shape[0])
        gamma_diag_str = f"{diag_mean:.4f}"
        gamma_off_str = f"{off_diag_mean:.4f}"
    else:
        diag_mean = 1.0 / gamma_final.shape[0] if gamma_final is not None else float('nan')
        gamma_diag_str = f"{diag_mean:.4f} (全均匀, Γ=I)"
        gamma_off_str = f"{diag_mean:.4f}"

    results = {
        "Experiment": name,
        "Γ_Enabled": not no_gamma,
        "Val_AUC": round(val_auc, 4),
        "Test_AUC": round(test_auc, 4),
        "Test_Acc": round(test_acc, 4),
        "Train_Time_s": round(train_time, 1),
        "Γ_Diag_Mean": round(diag_mean, 4),
        "Γ_OffDiag_Mean": round(off_diag_mean if not no_gamma else 1.0 / gamma_final.shape[0], 4) if gamma_final is not None else 0,
    }

    print(f"\n  {name} 结果:")
    print(f"    验证 AUC: {val_auc:.4f}")
    print(f"    测试 AUC: {test_auc:.4f}")
    print(f"    测试 Acc: {test_acc:.4f}")
    print(f"    训练耗时: {train_time:.1f}s")
    print(f"    Γ 对角线均值: {gamma_diag_str}")
    print(f"    Γ 非对角线均值: {gamma_off_str}")

    return results, model


def main():
    print("=" * 60)
    print("  Γ 矩阵消融实验 — Cross-Relation Risk Propagation")
    print("=" * 60)

    # ── 加载数据（只加载一次，两组共用） ──
    print("\n[0] 加载数据...")
    t0 = time.time()
    data = load_csmar_data_v5()
    print(f"  数据加载耗时 {time.time() - t0:.1f}s")

    # 过滤边类型
    valid_edge_types = [et for et in config.EDGE_TYPES if et in data.edge_index_dict]
    if valid_edge_types != config.EDGE_TYPES:
        missing = set(config.EDGE_TYPES) - set(valid_edge_types)
        print(f"  注意: 以下边类型缺失，跳过: {missing}")

    all_results = []

    # ═══════════════════════════════════════════════════════
    # 实验组 A: Γ 开启（正常模式）
    # ═══════════════════════════════════════════════════════
    res_a, model_a = run_experiment(
        "实验组A: Γ可学习", no_gamma=False,
        data=data, edge_types=valid_edge_types)
    all_results.append(res_a)

    # ═══════════════════════════════════════════════════════
    # 对照组 B: Γ 关闭（消融模式）
    # ═══════════════════════════════════════════════════════
    res_b, model_b = run_experiment(
        "对照组B: Γ=I (消融)", no_gamma=True,
        data=data, edge_types=valid_edge_types)
    all_results.append(res_b)

    # ═══════════════════════════════════════════════════════
    # 对比总结
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  消融实验结果对比")
    print("=" * 60)

    df = pd.DataFrame(all_results)
    print(df.to_string(index=False))

    auc_diff = res_a["Test_AUC"] - res_b["Test_AUC"]
    print(f"\n  Δ Test AUC (Γ - 无Γ): {auc_diff:+.4f}")

    if auc_diff > 0.005:
        print(f"  → 结论: Γ 矩阵有显著正向贡献 (+{auc_diff:.4f} AUC)")
        print(f"  → 跨关系风险传播在异构图上是有效的")
    elif auc_diff > 0.001:
        print(f"  → 结论: Γ 矩阵有轻微正向贡献 (+{auc_diff:.4f} AUC)")
        print(f"  → 保留 Γ 有利于模型，但贡献有限")
    elif auc_diff > -0.001:
        print(f"  → 结论: Γ 矩阵基本无贡献 (|Δ| < 0.001)")
        print(f"  → 跨关系传播对当前数据无增量信号，Γ 可简化为恒等映射")
    else:
        print(f"  → 结论: Γ 矩阵有负面影响 ({auc_diff:.4f} AUC)")
        print(f"  → 跨关系传播引入噪声，建议移除")

    # 保存
    out_dir = config.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "ablation_gamma.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  结果已保存到 {csv_path}")


if __name__ == "__main__":
    main()
