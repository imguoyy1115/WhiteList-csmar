"""
================================================================================
简易异构图子图采样器（零额外依赖，替代 NeighborLoader）
================================================================================
对种子节点做 2-hop 邻居扩展 → 构建局部子图 → 返回模型可消费的 dict。
================================================================================
"""
import torch


def sample_subgraph(data, seed_ent_idx, num_hops=2):
    """
    ==========================================================================
    以 seed_ent_idx 为中心，扩展 num_hops 跳邻居，构建局部子图。

    参数:
      data:            HeteroGraphData（CPU）
      seed_ent_idx:    (B,) long tensor — 种子 enterprise 节点的全局 ID
      num_hops:        扩展跳数（2 层 SAGE 需要 2 跳）

    返回:
      batch: dict，包含 x_dict, edge_index_dict, seed_mask, n_ent, nid
    ==========================================================================
    """
    device = seed_ent_idx.device
    ei_all = data.edge_index_dict

    # ── 区分 enterprise↔enterprise 和跨类型边 ──
    ent_etypes = []   # (src='enterprise', rel, dst='enterprise')
    cross_etypes = [] # 其他
    for etype in ei_all:
        if etype[0] == "enterprise" and etype[2] == "enterprise":
            ent_etypes.append(etype)
        else:
            cross_etypes.append(etype)

    # ── 2-hop 扩张（只用 enterprise↔enterprise 边） ──
    current = seed_ent_idx
    all_ent = seed_ent_idx  # 累积所有 enterprise 节点

    for _ in range(num_hops):
        neighbors = []
        for etype in ent_etypes:
            ei = ei_all[etype]  # [2, E]
            mask = torch.isin(ei[0], current) | torch.isin(ei[1], current)
            if mask.any():
                neighbors.append(ei[0, mask])
                neighbors.append(ei[1, mask])
        if not neighbors:
            break
        new_neighbors = torch.cat(neighbors).unique()
        # 去掉已经有的
        existing_mask = torch.isin(new_neighbors, all_ent)
        current = new_neighbors[~existing_mask]
        if current.numel() == 0:
            break
        all_ent = torch.cat([all_ent, current])

    # 排序 + 全局→局部映射
    all_ent = all_ent.unique()
    ent_map = torch.full((data.num_enterprises,), -1, dtype=torch.long, device=device)
    ent_map[all_ent] = torch.arange(len(all_ent), device=device)

    # ── 构建 x_dict ──
    x_dict_batch = {"enterprise": data.x_dict["enterprise"][all_ent]}

    # ── 构建 edge_index_dict ──
    edge_index_dict_batch = {}

    # enterprise↔enterprise 边
    for etype in ent_etypes:
        ei = ei_all[etype]
        src_mask = ent_map[ei[0]] >= 0
        dst_mask = ent_map[ei[1]] >= 0
        edge_mask = src_mask & dst_mask
        if edge_mask.any():
            edge_index_dict_batch[etype] = torch.stack([
                ent_map[ei[0, edge_mask]], ent_map[ei[1, edge_mask]]
            ])

    # 跨类型边（enterprise→riskevent 等）
    for etype in cross_etypes:
        ei = ei_all[etype]
        src_mask = ent_map[ei[0]] >= 0
        if not src_mask.any():
            continue
        dst_all = ei[1, src_mask].unique()
        dst_list = dst_all.sort()[0]
        dst_map = torch.full(
            (max(dst_list.max().item() + 1, 1),), -1,
            dtype=torch.long, device=device
        )
        dst_map[dst_list] = torch.arange(len(dst_list), device=device)

        # 添加目标节点特征
        dst_type = etype[2]
        if dst_type in data.x_dict:
            x_dict_batch[dst_type] = data.x_dict[dst_type][dst_list]

        local_src = ent_map[ei[0, src_mask]]
        local_dst = dst_map[ei[1, src_mask]]
        edge_index_dict_batch[etype] = torch.stack([local_src, local_dst])

    # ── 种子节点的局部 ID（用于 loss 计算） ──
    seed_local = ent_map[seed_ent_idx]
    seed_mask = torch.zeros(len(all_ent), dtype=torch.bool, device=device)
    seed_mask[seed_local] = True

    return {
        "x_dict": x_dict_batch,
        "edge_index_dict": edge_index_dict_batch,
        "seed_mask": seed_mask,          # 哪些 enterprise 节点是本次的种子
        "n_ent": len(all_ent),
        "nid": all_ent,                  # 全局 enterprise ID（用于索引 x_struct 等）
    }
