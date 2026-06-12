"""
================================================================================
假数据构建器 — 模拟赛题数据格式
================================================================================
产出的 HeteroGraphData 接口跟真实数据完全一致。
数据到手后：删掉本文件，写一个 load_real_data() 返回同样的 dataclass。
================================================================================
"""

import numpy as np
import torch
from data_interface import HeteroGraphData
from config import (
    SYNTH_N_ENT, SYNTH_N_PERSON, SYNTH_N_RISK, SYNTH_N_PROJ,
    SYNTH_DIM_ENT, SYNTH_DIM_PERSON, SYNTH_DIM_RISK, SYNTH_DIM_PROJ,
    SYNTH_N_MONTHS, SEED, EDGE_TYPE_NAMES,
)


def build_synthetic_data() -> HeteroGraphData:
    """
    ==========================================================================
    构建假数据，模拟真实赛题数据的规模和结构。
    返回 HeteroGraphData，跟 load_real_data() 的接口完全一致。
    ==========================================================================
    """
    np.random.seed(SEED)

    # ---- 1. 节点特征 ----
    x_dict = {
        "enterprise": torch.randn(SYNTH_N_ENT, SYNTH_DIM_ENT).float(),
        "person":     torch.randn(SYNTH_N_PERSON, SYNTH_DIM_PERSON).float(),
        "riskevent":  torch.randn(SYNTH_N_RISK, SYNTH_DIM_RISK).float(),
        "project":    torch.randn(SYNTH_N_PROJ, SYNTH_DIM_PROJ).float(),
    }

    # ---- 2. 构建边（模拟真实的度分布） ----
    edge_index_dict = {}
    n_ent, n_per, n_risk, n_proj = SYNTH_N_ENT, SYNTH_N_PERSON, SYNTH_N_RISK, SYNTH_N_PROJ

    # ---- 辅助函数：确保每个节点类型至少作为一次目的节点 ----
    def _add_rev(edge_index_dict, src_type, etype, dst_type, src, dst):
        """添加双向边：正向 + 反向（反向边类型名加 _rev 后缀）"""
        edge_index_dict[(src_type, etype, dst_type)] = (
            torch.tensor(np.stack([src, dst])).long()
        )
        # 反向边
        rev_etype = etype + "_rev"
        edge_index_dict[(dst_type, rev_etype, src_type)] = (
            torch.tensor(np.stack([dst, src])).long()
        )

    # trade: 企业间交易边
    n_trade = n_ent * 3
    src = np.random.randint(0, n_ent, n_trade)
    dst = np.random.randint(0, n_ent, n_trade)
    mask = src != dst
    src, dst = src[mask], dst[mask]
    edge_index_dict[("enterprise", "trade", "enterprise")] = (
        torch.tensor(np.stack([src, dst])).long()
    )
    edge_index_dict[("enterprise", "trade_rev", "enterprise")] = (
        torch.tensor(np.stack([dst, src])).long()
    )

    # equity: 股权控制边
    n_equity = n_ent * 2
    src = np.random.choice(n_ent, n_equity, p=_powerlaw_weights(n_ent))
    dst = np.random.randint(0, n_ent, n_equity)
    src = np.clip(src, 0, n_ent-1)
    edge_index_dict[("enterprise", "equity", "enterprise")] = (
        torch.tensor(np.stack([src, dst])).long()
    )
    edge_index_dict[("enterprise", "equity_rev", "enterprise")] = (
        torch.tensor(np.stack([dst, src])).long()
    )

    # legal_rep: 法人→企业（+ 反向边使 person 也能收到消息）
    n_legal = n_ent
    src = np.random.randint(0, n_per, n_legal)
    dst = np.random.randint(0, n_ent, n_legal)
    edge_index_dict[("person", "legal_rep", "enterprise")] = (
        torch.tensor(np.stack([src, dst])).long()
    )
    edge_index_dict[("enterprise", "legal_rep_rev", "person")] = (
        torch.tensor(np.stack([dst, src])).long()
    )

    # involved_in: 企业→风险事件（+ 反向边）
    n_inv = n_ent // 3
    src = np.random.randint(0, n_ent, n_inv)
    dst = np.random.randint(0, n_risk, n_inv)
    edge_index_dict[("enterprise", "involved_in", "riskevent")] = (
        torch.tensor(np.stack([src, dst])).long()
    )
    edge_index_dict[("riskevent", "involved_in_rev", "enterprise")] = (
        torch.tensor(np.stack([dst, src])).long()
    )

    # same_industry: 同行业（无向）
    n_si = n_ent // 5
    src = np.random.randint(0, n_ent, n_si)
    dst = np.random.randint(0, n_ent, n_si)
    edge_index_dict[("enterprise", "same_industry", "enterprise")] = (
        torch.tensor(np.stack([np.concatenate([src, dst]),
                                np.concatenate([dst, src])])).long()
    )

    # bid: 企业→招投标（+ 反向边）
    n_bid = n_ent // 4
    src = np.random.randint(0, n_ent, n_bid)
    dst = np.random.randint(0, n_proj, n_bid)
    edge_index_dict[("enterprise", "bid", "project")] = (
        torch.tensor(np.stack([src, dst])).long()
    )
    edge_index_dict[("project", "bid_rev", "enterprise")] = (
        torch.tensor(np.stack([dst, src])).long()
    )

    # ---- 3. 标签 ----
    # 白名单标签：负面信号少的企业 = 正样本
    # 模拟：涉诉少 + 交易稳定 → 白名单
    involved_count = np.bincount(
        edge_index_dict[("enterprise", "involved_in", "riskevent")][0].numpy(),
        minlength=n_ent
    )
    y_white = (involved_count <= np.median(involved_count)).astype(np.float32)
    y_white = torch.tensor(y_white)
    # 风险标签（t+3）：部分白名单企业未来也会出问题
    y_risk = torch.tensor((np.random.rand(n_ent) < 0.3).astype(np.float32))
    # 五级分层：从风险+白名单推导
    grade = np.zeros(n_ent, dtype=int)
    grade[(y_white == 1) & (y_risk == 0)] = 0   # S
    grade[(y_white == 1) & (y_risk == 1)] = 1   # A
    grade[(y_white == 0) & (y_risk == 0)] = 2   # B
    grade[(y_white == 0) & (y_risk == 1)] = 3   # C
    y_grade = torch.tensor(grade)

    # ---- 4. Mask（模拟时间切分：前 60% 训练，中间 20% 验证，后 20% 测试） ----
    # 假数据没有时间维度，用随机切分近似
    perm = np.random.permutation(n_ent)
    n_train = int(n_ent * 0.6)
    n_val = int(n_ent * 0.2)
    train_mask = torch.zeros(n_ent, dtype=bool); train_mask[perm[:n_train]] = True
    val_mask   = torch.zeros(n_ent, dtype=bool); val_mask[perm[n_train:n_train+n_val]] = True
    test_mask  = torch.zeros(n_ent, dtype=bool); test_mask[perm[n_train+n_val:]] = True

    # ---- 5. 图结构指标（Layer 5 用） ----
    degree = torch.zeros(n_ent)
    for etype in [("enterprise","trade","enterprise"), ("enterprise","equity","enterprise")]:
        ei = edge_index_dict.get(etype)
        if ei is not None:
            degree += torch.bincount(ei[0], minlength=n_ent)
    graph_metrics = {
        "pagerank": torch.rand(n_ent),        # 假数据用随机数占位
        "betweenness": torch.rand(n_ent),
        "kcore": torch.randint(1, 6, (n_ent,)),
        "degree": degree,
        "core_distance": torch.rand(n_ent),
    }

    # ---- 6. 政策标签 ----
    policy_tags = torch.tensor((np.random.rand(n_ent) < 0.2).astype(np.float32))

    return HeteroGraphData(
        x_dict=x_dict,
        edge_index_dict=edge_index_dict,
        y_white=y_white,
        y_risk=y_risk,
        y_grade=y_grade,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_enterprises=n_ent,
        num_months=SYNTH_N_MONTHS,
        graph_metrics=graph_metrics,
        policy_tags=policy_tags,
    )


def _powerlaw_weights(n: int) -> np.ndarray:
    """生成幂律分布权重，模拟真实股权集中度（少数节点拥有大量边）"""
    w = 1.0 / (np.arange(n) + 1)
    return w / w.sum()
