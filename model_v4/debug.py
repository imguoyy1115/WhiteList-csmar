"""
================================================================================
CSMAR MODEL DIAGNOSTIC TOOL v4 (FIXED FOR CURRENT LOADER)
================================================================================
适配：
✔ load_all()
✔ build_entities / build_features / build_edges pipeline
✔ load_csmar_data()（推荐入口）
================================================================================
"""

import sys
import os
import time
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# =========================
# 1. 正确导入（已修复）
# =========================
from data_loader.csmar_loader import load_csmar_data, load_all

MODEL_PATH = "outputs/model_v4_csmar.pt"


# =========================
# 2. 基础检查函数
# =========================
def check_data_integrity(data):
    print("\n================ DATA CHECK ================")

    print("num_enterprises:", data.num_enterprises)

    print("\nx_dict keys:", list(data.x_dict.keys()))
    for k, v in data.x_dict.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    print("\nedge_index_dict:")
    for k, v in data.edge_index_dict.items():
        print(f"  {k}: shape={v.shape}")

    print("\nlabels:")
    print("  y_white:", data.y_white.shape)
    print("  y_risk :", data.y_risk.shape)
    print("  y_grade:", data.y_grade.shape)

    print("\nmasks:")
    print("  train:", data.train_mask.sum().item())
    print("  val  :", data.val_mask.sum().item())
    print("  test :", data.test_mask.sum().item())


# =========================
# 3. 模型文件检查
# =========================
def check_model():
    print("\n================ MODEL CHECK ================")

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        return False

    try:
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        print("✔ model loaded successfully")

        if isinstance(ckpt, dict):
            print("keys:", ckpt.keys())

        return True

    except Exception as e:
        print("[ERROR] failed to load model:", e)
        return False


# =========================
# 4. 结构一致性检查
# =========================
def check_feature_consistency(data):
    print("\n================ FEATURE CHECK ================")

    x = data.x_dict["enterprise"]

    print("feature shape:", x.shape)

    # 检查是否全零（常见致命问题）
    zero_ratio = (x.sum(dim=1) == 0).float().mean().item()
    print(f"zero-feature ratio: {zero_ratio:.4f}")

    if zero_ratio > 0.5:
        print("⚠ WARNING: too many zero features!")

    print("feature stats:")
    print("  mean:", x.mean().item())
    print("  std :", x.std().item())


# =========================
# 5. 标签分布检查
# =========================
def check_labels(data):
    print("\n================ LABEL CHECK ================")

    yw = data.y_white.numpy()
    yr = data.y_risk.numpy()
    yg = data.y_grade.numpy()

    print("white=1 ratio:", yw.mean())
    print("risk =1 ratio:", yr.mean())

    print("grade distribution:")
    for i in range(4):
        print(f"  class {i}: {(yg == i).mean():.4f}")


# =========================
# 6. edge检查
# =========================
def check_edges(data):
    print("\n================ EDGE CHECK ================")

    for k, ei in data.edge_index_dict.items():
        src, dst = ei
        print(f"\n{k}")
        print("  edges:", ei.shape[1])
        print("  src range:", (src.min().item(), src.max().item()))
        print("  dst range:", (dst.min().item(), dst.max().item()))


# =========================
# 7. main
# =========================
def main():
    print("\n================================================")
    print("  CSMAR MODEL DIAGNOSTIC TOOL v4 (FIXED)")
    print("================================================\n")

    t0 = time.time()

    # =====================
    # ✔ 正确入口（关键修复点）
    # =====================
    data = load_csmar_data()

    print(f"\n[OK] data loaded in {time.time()-t0:.2f}s")

    # =====================
    # diagnostics
    # =====================
    check_data_integrity(data)
    check_feature_consistency(data)
    check_labels(data)
    check_edges(data)
    check_model()

    print("\n================================================")
    print("DIAGNOSTIC COMPLETE")
    print("================================================")


if __name__ == "__main__":
    main()