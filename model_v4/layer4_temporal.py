"""
================================================================================
Layer 4: 时序预测层（GRU over snapshot embeddings）
================================================================================
输入: h_fusion = [h_v || h_risk]  的 12 个月序列
输出: Z_v → 用于 t+1 白名单预测 + t+3 风险预测 + 企业分级

当前假数据版本：没有真实时序数据，用 h_fusion 模拟 12 个月序列
数据到手后：每个月跑一次 Layer 2+3 → 堆叠 12 个月的 h_fusion → 喂 GRU
================================================================================
"""

import torch
import torch.nn as nn
from config import GRU_HIDDEN, GRU_LAYERS, DROPOUT


class TemporalEncoder(nn.Module):
    """
    ==========================================================================
    时序 GRU 编码器

    当前版本：静态图模拟（重复当前月 h_fusion 12 次）
    真实版本：12 个月的 h_fusion(t-12..t) 序列 → GRU → Z_v
    ==========================================================================
    """
    def __init__(self, input_dim: int, hidden: int = GRU_HIDDEN,
                 num_layers: int = GRU_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_fusion_current: torch.Tensor,
                h_fusion_seq: torch.Tensor = None,
                batch_size: int = 1024) -> torch.Tensor:
        """
        ==========================================================================
        输入:
          h_fusion_current: (N, input_dim)      当前月的融合表示
          h_fusion_seq:     (N, seq_len, input_dim)  12 个月历史序列（可选）
          batch_size:       分批处理大小（避免 OOM，默认 1024）

        当前假数据版本：
          没有真实时序数据，用当前月重复 12 次模拟

        数据到手后：
          每个月独立跑 Layer 2+3 → 堆叠 12 个 h_fusion → 传入 h_fusion_seq

        分批 GRU：
          每批独立跑 GRU → 取最后时间步 → 拼接，结果等价于全量跑，
          但内存从 O(N) 降到 O(batch_size)
        ==========================================================================
        """
        N = (h_fusion_seq.shape[0] if h_fusion_seq is not None
             else h_fusion_current.shape[0])
        z_parts = []

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)

            if h_fusion_seq is not None:
                batch_seq = h_fusion_seq[start:end]          # (B, 12, hidden)
            else:
                # 假数据模式：逐批 expand，不在内存中展开 (N, 12, hidden) 全量
                batch_cur = h_fusion_current[start:end]      # (B, hidden)
                batch_seq = batch_cur.unsqueeze(1).repeat(1, 12, 1)  # (B, 12, hidden)

            gru_out, _ = self.gru(batch_seq)                 # (B, 12, hidden)
            z_parts.append(gru_out[:, -1, :])                # (B, hidden)

        z_v = self.dropout(torch.cat(z_parts, dim=0))        # (N, hidden)
        return z_v
