"""
================================================================================
多任务预测头 — 白名单分类 + 风险预测 + 企业分级
================================================================================
输入：Layer 4 的 Z_v (GRU 时序编码)
输出：三个预测概率

v4.1 改动：
  - 分批推理：对大图自动切 chunk，避免全量 (N, 1/5) 的中间张量撑爆显存
================================================================================
"""

import torch
import torch.nn as nn
from config import HEAD_BATCH_SIZE, GRADE_CLASSES, DROPOUT


class MultiTaskHeads(nn.Module):
    """
    ==========================================================================
    三个并行的 MLP 预测头
    共享输入 Z_v，各自独立参数，支持分批推理防 OOM
    ==========================================================================
    """
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = DROPOUT):
        super().__init__()
        self.default_batch = HEAD_BATCH_SIZE

        # 白名单预测头（二分类）
        self.white_head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # 风险预测头（二分类，t+3）
        self.risk_head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # 企业分级头（五分类：S/A/B/C/D）
        self.grade_head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, GRADE_CLASSES),
        )

    def forward(self, z_v: torch.Tensor, batch_size: int = None):
        """
        输入: z_v (N, in_dim)
        输出:
          logit_white: (N, 1)   白名单 logits
          logit_risk:  (N, 1)   风险 logits
          logit_grade: (N, 5)   分级 logits

        小图直接跑，大图自动切 chunk。
        """
        bs = batch_size if batch_size is not None else self.default_batch
        N = z_v.size(0)

        if N <= bs:
            return (
                self.white_head(z_v),
                self.risk_head(z_v),
                self.grade_head(z_v),
            )

        white_out, risk_out, grade_out = [], [], []
        for start in range(0, N, bs):
            z = z_v[start:start + bs]
            white_out.append(self.white_head(z))
            risk_out.append(self.risk_head(z))
            grade_out.append(self.grade_head(z))

        return (
            torch.cat(white_out, dim=0),
            torch.cat(risk_out, dim=0),
            torch.cat(grade_out, dim=0),
        )
