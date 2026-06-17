#!/usr/bin/env python3
"""Generate Fig. 6a community-partition network panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from infomap import Infomap


_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / "data" / "topology"
FIG_DIR = _ROOT / "figures" / "topology"
MODEL_ORDER = ("FRP", "MOp", "Average", "Vanilla")


def load_wrec(model_path: Path) -> np.ndarray:
    checkpoint = torch.load(model_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    for key in ("wrec", "module.wrec"):
        if isinstance(state, dict) and key in state:
            return state[key].detach().cpu().numpy().astype(float)
    raise KeyError(f"No recurrent weight tensor found in {model_path}")


def load_models(model_dir: Path) -> list[tuple[str, np.ndarray]]:
    models = []
    for path in sorted(model_dir.glob("*.pt")):
        models.append((path.stem.split("_", 1)[0], load_wrec(path)))
        print(f"Loaded model: {path.name}")
    return models


def remove_isolated_nodes_like(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix)
    row_zero = np.max(np.abs(mat), axis=1) == 0
    col_zero = np.max(np.abs(mat), axis=0) == 0
    keep = ~(row_zero & col_zero)
    return mat[np.ix_(keep, keep)]


def binarize_directed(weight: np.ndarray) -> np.ndarray:
    matrix = np.asarray(weight, dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return (matrix != 0.0).astype(np.uint8)


def directed_modularity_gamma(weight: np.ndarray, labels: np.ndarray, gamma: float) -> float:
    labels = np.asarray(labels)
    mask = labels != -1
    if mask.sum() == 0:
        return 0.0
    adjacency = (np.asarray(weight) != 0).astype(float)
    adjacency = adjacency[np.ix_(mask, mask)]
    edge_count = adjacency.sum()
    if edge_count <= 0:
        return 0.0
    out_degree = adjacency.sum(axis=1)
    in_degree = adjacency.sum(axis=0)
    communities = labels[mask]
    same = (communities[:, None] == communities[None, :]).astype(float)
    value = ((adjacency - gamma * np.outer(out_degree, in_degree) / edge_count) * same).sum() / edge_count
    return float(value)


def detect_communities(matrix: np.ndarray, n_runs: int = 1, seed_base: int | None = None, trials: int = 50, two_level: bool = True, gamma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    mat = np.asarray(matrix).copy().astype(float)
    np.fill_diagonal(mat, 0.0)
    node_count = mat.shape[0]
    binary = (mat != 0).astype(float)
    active = (binary.sum(axis=1) + binary.sum(axis=0)) > 0
    idx = np.where(active)[0] if active.any() else np.arange(node_count)
    sub = binary[np.ix_(idx, idx)]
    sub_count = len(idx)

    rows, cols = np.where(sub > 0)
    no_loop = rows != cols
    rows, cols = rows[no_loop], cols[no_loop]
    if rows.size == 0:
        return np.full((n_runs, node_count), -1, dtype=int), np.zeros(n_runs, dtype=float)

    base_flags = ["--directed", "--silent", f"-N {int(trials)}"]
    if two_level:
        base_flags.append("--two-level")

    labels_all = []
    scores = []
    for run_idx in range(n_runs):
        flags = base_flags.copy()
        if seed_base is not None:
            seed = max(1, int(seed_base + run_idx))
            flags += ["--seed", str(seed)]
        infomap = Infomap(" ".join(flags))
        add_edge = getattr(infomap, "add_link", None) or getattr(infomap, "addLink", None)
        for row, col in zip(rows, cols):
            add_edge(int(row), int(col), 1.0)
        infomap.run()

        sub_labels = np.full(sub_count, -1, dtype=int)
        for node in infomap.nodes:
            sub_labels[node.node_id] = node.module_id
        labels = np.full(node_count, -1, dtype=int)
        labels[idx] = sub_labels
        labels_all.append(labels)
        scores.append(directed_modularity_gamma(sub, sub_labels, gamma))

    return np.vstack(labels_all), np.asarray(scores, dtype=float)


def run_infomap_repeats(adjacency: np.ndarray, n_runs: int = 1, seed_base: int = 1, trials: int = 50, two_level: bool = True) -> tuple[np.ndarray, float, list[float]]:
    labels_all = []
    scores = []
    for offset in range(int(n_runs)):
        labels_matrix, score_array = detect_communities(
            adjacency,
            n_runs=1,
            seed_base=int(seed_base) + offset,
            trials=int(trials),
            two_level=bool(two_level),
        )
        labels_all.append(np.asarray(labels_matrix[0]).reshape(-1))
        scores.append(float(np.asarray(score_array).ravel()[0]))
    best_idx = int(np.argmax(scores)) if scores else 0
    return labels_all[best_idx], float(scores[best_idx]), list(map(float, scores))

def plot_network(
    sc_thr,
    labels,
    save_path=None,
    show_labels=False,
    figsize=(8, 8),
    edge_alpha=0.8,
    drop_unassigned=True,         # True: 不绘制 label=-1 节点
    unassigned_color="#C0C0C0",   # -1 节点颜色
    symmetrize=None,              # 'mean'|'max'|'sum'|None  有向->无向时的对称化；None=保留有向
    weight_mode='positive',       # 'positive'(仅>0) | 'abs'(取绝对值) | 'raw'(原始)
    min_weight=0.0,               # 丢弃“很弱”边：'positive'基于 w，其它基于 |w|
    remove_isolates=True,         # True: 移除孤立点
    layout='community',           # 'community' | 'spring'
    random_state=42,

    # —— 高亮配置（仅对“跨社区边”打分并用黑色高亮） —— #
    # highlight_top=0.10,           # 前 10%
    highlight_top= 0.20,           # 前 20%
    highlight_width_add=1.1,      # 高亮线宽加成（绝对）
    highlight_alpha=1.0,          # 高亮透明度

    # —— 组间对比增强 —— #
    cross_base_color="#C0C0C0",   # 普通组间边颜色（浅灰）
    cross_highlight_color="black",# 高亮边颜色（跨社区黑色）
    cross_base_alpha=0.5,         # None -> 用 edge_alpha
    hl_width_factor=1.1,          # 高亮线宽至少是普通线宽的倍数
    hl_width_min=1.1,             # 高亮线宽绝对下限

    # —— 社区数量控制（超限时合并为 bundle） —— #
    max_visual_communities: int = 15,
    merged_bundles: int = 1,

    # —— 箭头与短边避让（FancyArrowPatch 专用，仅在有向模式下生效） —— #
    arrow_base=6.0,               # 箭头基准尺寸（pt）
    arrow_scale=2.2,              # 箭头随线宽放大
    margin_pad=2.5,               # 箭头与节点外沿的额外边距（pt）
    arc_rad=0.12,                 # 互易边弧度
):
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, FancyArrowPatch

    # ---------------- 工具：安全设置 zorder（兼容旧版 networkx 返回对象） ----------------
    def _safe_set_zorder(coll, z):
        try:
            if isinstance(coll, (list, tuple)):
                for it in coll:
                    try: it.set_zorder(z)
                    except Exception: pass
            else:
                coll.set_zorder(z)
        except Exception:
            pass

    # 将数据坐标长度换成像素长度
    def _edge_pix_len(u, v, pos, ax):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        p0 = ax.transData.transform((x0, y0))
        p1 = ax.transData.transform((x1, y1))
        return float(np.hypot(p1[0]-p0[0], p1[1]-p0[1]))

    # 1) 预处理
    W = np.array(sc_thr, dtype=float, copy=True)
    if W.shape[0] != W.shape[1]:
        raise ValueError("sc_thr 必须是方阵")
    np.fill_diagonal(W, 0.0)

    # —— 对称化（若 symmetrize=None 则不对称化，保留有向） —— #
    if symmetrize is not None and not np.allclose(W, W.T, equal_nan=True):
        if symmetrize == 'mean':
            W = 0.5 * (W + W.T)
        elif symmetrize == 'max':
            W = np.maximum(W, W.T)
        elif symmetrize == 'sum':
            W = W + W.T
        else:
            raise ValueError("symmetrize 必须为 'mean' | 'max' | 'sum' | None")

    # —— 权重处理（保持原语义） —— #
    if weight_mode == 'positive':
        W = np.where(W > 0, W, 0.0)
        if min_weight > 0: W = np.where(W > min_weight, W, 0.0)
    elif weight_mode == 'abs':
        W = np.abs(W)
        if min_weight > 0: W = np.where(W > min_weight, W, 0.0)
    elif weight_mode == 'raw':
        if min_weight > 0: W = np.where(np.abs(W) > min_weight, W, 0.0)
    else:
        raise ValueError("weight_mode 必须为 'positive' | 'abs' | 'raw'")

    # 2) 建图：symmetrize=None → 有向；否则无向
    N = W.shape[0]
    directed_mode = (symmetrize is None)
    if directed_mode:
        G = nx.DiGraph()
        G.add_nodes_from(range(N))
        r, c = np.where(W != 0.0)
        mask = (r != c)
        for i, j in zip(r[mask], c[mask]):
            G.add_edge(int(i), int(j), weight=float(W[i, j]))
    else:
        G = nx.Graph()
        G.add_nodes_from(range(N))
        r, c = np.where(np.triu(W, k=1) != 0.0)
        for i, j in zip(r, c):
            G.add_edge(int(i), int(j), weight=float(W[i, j]))

    if remove_isolates:
        G.remove_nodes_from(list(nx.isolates(G)))
    if G.number_of_edges() == 0:
        print("Graph empty after preprocessing."); return

    # 3) 标签
    labs = np.asarray(labels)
    if labs.shape[0] != N:
        raise ValueError("labels 长度必须与矩阵规模一致")

    nodes = list(G.nodes())
    node_to_label = {n: int(labs[n]) for n in nodes}

    if drop_unassigned:
        keep_nodes = [n for n in nodes if node_to_label[n] != -1]
        G = G.subgraph(keep_nodes).copy()
        if G.number_of_edges() == 0:
            print("Only unassigned (-1) nodes remained or no edges."); return
        nodes = list(G.nodes())
        node_to_label = {n: int(labs[n]) for n in nodes}

    # —— 节点度和：有向=入度+出度；无向=度 —— #
    if isinstance(G, nx.DiGraph):
        _in_deg  = dict(G.in_degree())
        _out_deg = dict(G.out_degree())
        deg_sum_map = {n: _in_deg.get(n, 0) + _out_deg.get(n, 0) for n in G.nodes()}
    else:
        deg_sum_map = dict(G.degree())
# ---------------- 新增：跳过大小为1或2的社区（可配置） ----------------
    skip_size_threshold = 2  # 小于此大小的社区将被跳过（即 1 和 2 会被跳过）
# ######################################################################
    # 计算每个社区的成员节点（基于当前 node_to_label 与 nodes 列表）
    from collections import defaultdict
    label_to_nodes = defaultdict(list)
    for n in nodes:
        lbl = node_to_label.get(n, -1)
        if lbl == -1: continue
        label_to_nodes[lbl].append(n)

    # 找出应当跳过的社区标签
    skip_labels = {lbl for lbl, members in label_to_nodes.items() if len(members) < skip_size_threshold}

    if skip_labels:
        # 从 nodes 中移除这些社区内的节点（以及它们的 edges 会自动被排除）
        keep_nodes = [n for n in nodes if node_to_label.get(n, -1) not in skip_labels]
        if not keep_nodes:
            print("所有节点都属于被跳过的社区，绘图终止。")
            return

        # 重新取子图并更新相关结构
        G = G.subgraph(keep_nodes).copy()
        nodes = list(G.nodes())
        # 注意：这里仍保留原 labels 数组（labs），但 node_to_label 需重建
        node_to_label = {n: int(labs[n]) for n in nodes}

        # 重新计算 labels_all / uniq_labels / order_labels 等（用于颜色映射和后续逻辑）
        labels_all = [node_to_label[n] for n in nodes if node_to_label[n] != -1]
        uniq_labels = sorted(set(labels_all))
        order_labels = uniq_labels

# #######################################################
    # 3.5) 社区数量超限 → 合并为 bundle（保持原逻辑）
    labels_all = [node_to_label[n] for n in nodes if node_to_label[n] != -1]
    uniq_labels = sorted(set(labels_all))
    keep_set = set(uniq_labels)
    bundle_labels = []
    if len(uniq_labels) > max_visual_communities:
        from collections import Counter, defaultdict
        size_counter = Counter(labels_all)
        intra_strength = defaultdict(float)
        inter_strength = defaultdict(lambda: defaultdict(float))
        for u, v, d in G.edges(data=True):
            lu, lv = node_to_label[u], node_to_label[v]
            w = abs(d.get('weight', 0.0))
            if lu == -1 or lv == -1: continue
            if lu == lv:
                intra_strength[lu] += w
            else:
                a, b = (lu, lv) if lu < lv else (lv, lu)
                inter_strength[a][b] += w
        k_bundles = max(1, int(merged_bundles))
        k_bundles = min(k_bundles, max_visual_communities - 1)
        keep_count = max_visual_communities - k_bundles
        ordered = sorted(uniq_labels, key=lambda g: (intra_strength[g], size_counter[g]), reverse=True)
        keep_set = set(ordered[:keep_count])
        weak = [g for g in ordered[keep_count:]]
        bundles = [set() for _ in range(k_bundles)]
        def inter_w(a, b):
            x, y = (a, b) if a < b else (b, a)
            return inter_strength[x].get(y, 0.0)
        for g in weak:
            best_i, best_gain = 0, -1.0
            for i in range(k_bundles):
                gain = sum(inter_w(g, h) for h in bundles[i]) if bundles[i] else 0.0
                if gain > best_gain:
                    best_i, best_gain = i, gain
            bundles[best_i].add(g)
        base_new = (max(uniq_labels) + 1) if uniq_labels else 0
        bundle_labels = [base_new + i for i in range(k_bundles)]
        g2bundle = {}
        for i, S in enumerate(bundles):
            for g in S: g2bundle[g] = bundle_labels[i]
        for n in nodes:
            g = node_to_label[n]
            if g != -1 and g not in keep_set:
                node_to_label[n] = g2bundle[g]
        uniq_labels = sorted(set([node_to_label[n] for n in nodes if node_to_label[n] != -1]))

    
    # 4) 颜色与大小
    order_labels = uniq_labels
    base_cmap = plt.get_cmap('tab20')
    color_map = {lbl: base_cmap(i % getattr(base_cmap, 'N', 20)) for i, lbl in enumerate(order_labels)}
    node_colors = [color_map.get(node_to_label[n], unassigned_color) for n in nodes]

    strength = dict(G.degree(weight='weight'))
    s_vals = np.array([strength[n] for n in nodes], float)
    smin, smax = float(s_vals.min()), float(s_vals.max()); sden = (smax - smin) if smax > smin else 1.0
    node_sizes = [40 + 100 * (strength[n] - smin) / sden for n in nodes]

    edges_all = list(G.edges(data=True))
    w_abs = np.array([abs(d['weight']) for _, _, d in edges_all], float)
    wmin, wmax = float(w_abs.min()), float(w_abs.max()); wden = (wmax - wmin) if wmax > wmin else 1.0
    def width_map(w, gamma=0.8):
        if wden <= 0: return 2.0
        x = (abs(w) - wmin) / wden
        x = np.clip(x, 0, 1) ** gamma
        return 1 + 5 * x

    # 5) 布局
    rng = np.random.RandomState(random_state)
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=random_state, weight='weight')
    elif layout == 'community':
        groups = [[n for n in nodes if node_to_label[n] == g] for g in order_labels]
        pos = {}
        K = len(groups)
        ang = np.linspace(0, 2*np.pi, num=max(K,1), endpoint=False)
        R_outer = 6.0
        centers = [(R_outer*np.cos(a), R_outer*np.sin(a)) for a in ang] if K>0 else [(0.0,0.0)]
        for (gnodes, center) in zip(groups, centers):
            if not gnodes: continue
            sub = G.subgraph(gnodes)
            k = 1.0 / np.sqrt(max(len(gnodes), 1))
            sub_pos = nx.spring_layout(sub, seed=rng, weight='weight', k=k, iterations=50)
            xs = np.array([p[0] for p in sub_pos.values()]); ys = np.array([p[1] for p in sub_pos.values()])
            if np.ptp(xs)==0: xs = xs + rng.randn(*xs.shape)*1e-3
            if np.ptp(ys)==0: ys = ys + rng.randn(*ys.shape)*1e-3
            xs = (xs - xs.mean())/(np.ptp(xs)+1e-9); ys = (ys - ys.mean())/(np.ptp(ys)+1e-9)
            scale = 1.4
            for n, (x, y) in zip(sub_pos.keys(), zip(xs*scale, ys*scale)):
                pos[n] = (center[0] + x, center[1] + y)
    else:
        raise ValueError("layout 必须为 'community' 或 'spring'")

    # 6) 画边（全部先画完；节点最后画）
    fig, ax = plt.subplots(figsize=figsize)

    # 分类边
    cross_edges, cross_w_base = [], []
    same_edges = {lbl: [] for lbl in order_labels}
    same_w_base = {lbl: [] for lbl in order_labels}
    base_width_cache = {}
    for u, v, d in edges_all:
        w = d['weight']; lu, lv = node_to_label[u], node_to_label[v]
        w_base = width_map(w); base_width_cache[(u, v)] = w_base
        if lu == -1 or lv == -1 or lu != lv:
            cross_edges.append((u, v)); cross_w_base.append(w_base)
        else:
            same_edges[lu].append((u, v)); same_w_base[lu].append(w_base)

    # —— 仅对“跨社区边”打分并选前 p%（按两端节点的入+出度之和）——
    edge_key = (lambda u, v: (u, v)) if directed_mode else (lambda u, v: (min(u, v), max(u, v)))
    hl_cross_set = set()
    hl_cross, hlw_cross = [], []

    scored = []
    for u, v, d in edges_all:
        lu, lv = node_to_label[u], node_to_label[v]
        if lu == -1 or lv == -1 or lu == lv:
            continue  # 仅考虑跨社区
        score = deg_sum_map.get(u, 0) + deg_sum_map.get(v, 0)
        scored.append((score, u, v, d.get('weight', 1.0)))

    scored.sort(reverse=True, key=lambda x: x[0])
    k_top = max(1, int(np.ceil(highlight_top * len(scored))))
    top = scored[:k_top] if scored else []

    for _, u, v, w in top:
        base_w = base_width_cache.get((u, v), width_map(w))
        w_hl = max(base_w * hl_width_factor, base_w + highlight_width_add, hl_width_min)
        ekey = edge_key(u, v)
        if ekey in hl_cross_set:  # 避免重复
            continue
        hl_cross.append((u, v))
        hlw_cross.append(float(w_hl))
        hl_cross_set.add(ekey)

    # ---------- 有向 & 无向分别绘制 ----------
    if not directed_mode:
        # 无向基础：组间（灰）
        if cross_edges:
            base_filtered, base_w_filtered = [], []
            for (e, w) in zip(cross_edges, cross_w_base):
                if edge_key(*e) not in hl_cross_set:
                    base_filtered.append(e); base_w_filtered.append(w)
            if base_filtered:
                coll = nx.draw_networkx_edges(
                    G, pos, edgelist=base_filtered,
                    width=[float(x) for x in base_w_filtered],
                    edge_color=cross_base_color,
                    alpha=edge_alpha if cross_base_alpha is None else cross_base_alpha, ax=ax
                ); _safe_set_zorder(coll, 1)
        # 无向基础：组内（社区色）
        for lbl in order_labels:
            if same_edges[lbl]:
                base_list, base_w = [], []
                for e, w in zip(same_edges[lbl], same_w_base[lbl]):
                    base_list.append(e); base_w.append(w)
                coll = nx.draw_networkx_edges(
                    G, pos, edgelist=base_list,
                    width=[float(x) for x in base_w],
                    edge_color=color_map[lbl],
                    alpha=min(edge_alpha+0.15, 1.0), ax=ax
                ); _safe_set_zorder(coll, 2)

        # —— 无向高亮：只画跨社区黑色（使用宽度映射确保对齐）—— #
        if hl_cross:
            hlw_map = {e: float(w) for e, w in zip(hl_cross, hlw_cross)}
            coll = nx.draw_networkx_edges(
                G, pos, edgelist=hl_cross,
                width=[hlw_map[e] for e in hl_cross],
                edge_color=cross_highlight_color,
                alpha=highlight_alpha, ax=ax
            ); _safe_set_zorder(coll, 3)

    else:
        # —— 有向模式：逐条边用 FancyArrowPatch，带 shrinkA/B + 动态限幅箭头 ——
        has_rev = set((v, u) for (u, v) in G.edges())

        # 计算节点半径（pt）与 shrink
        node_radius_pt = {n: (np.sqrt(node_sizes[nodes.index(n)]) / 2.0) for n in nodes}
        shrink_pt = {n: node_radius_pt[n] + float(margin_pad) for n in nodes}

        def _arrow_size_for(u, v, w_pt):
            # 基于线宽的名义尺寸
            nominal = float(arrow_base) + float(arrow_scale) * float(w_pt)
            # 边有效像素长度（扣去两端 shrink）
            pix_len = _edge_pix_len(u, v, pos, ax)
            eff = max(0.0, pix_len - (shrink_pt[u] + shrink_pt[v]))
            limit = 0.45 * eff  # 箭头头长上限 = 可见长度的 45%
            return max(1.0, min(nominal, limit))

        def _draw_dir_edges(edgelist, color, alpha, zorder=1, curved=False, widths=None):
            if not edgelist: return
            if widths is None:
                widths = [base_width_cache[(u, v)] for (u, v) in edgelist]
            for (u, v), w in zip(edgelist, widths):
                arr = _arrow_size_for(u, v, w)
                con = (f'arc3,rad={arc_rad}') if ((v, u) in has_rev and curved) else None
                patch = FancyArrowPatch(
                    posA=pos[u], posB=pos[v],
                    arrowstyle='-|>',
                    mutation_scale=arr,
                    linewidth=float(w),
                    color=color,
                    alpha=alpha,
                    shrinkA=shrink_pt[u],
                    shrinkB=shrink_pt[v],
                    connectionstyle=con,
                    zorder=zorder
                )
                ax.add_patch(patch)

        # 组间基础（灰）
        cross_base = [e for e in cross_edges if edge_key(*e) not in hl_cross_set]
        if cross_base:
            straight = [e for e in cross_base if (e[1], e[0]) not in has_rev]
            curved   = [e for e in cross_base if (e[1], e[0]) in  has_rev]
            _draw_dir_edges(straight, cross_base_color,
                            edge_alpha if cross_base_alpha is None else cross_base_alpha, zorder=1, curved=False)
            _draw_dir_edges(curved,   cross_base_color,
                            edge_alpha if cross_base_alpha is None else cross_base_alpha, zorder=1, curved=True)

        # 组内基础（各自颜色）
        for lbl in order_labels:
            if same_edges[lbl]:
                straight = [e for e in same_edges[lbl] if (e[1], e[0]) not in has_rev]
                curved   = [e for e in same_edges[lbl] if (e[1], e[0]) in  has_rev]
                a = min(edge_alpha+0.15, 1.0)
                _draw_dir_edges(straight, color_map[lbl], a, zorder=2, curved=False)
                _draw_dir_edges(curved,   color_map[lbl], a, zorder=2, curved=True)

        # —— 有向高亮：只画跨社区黑色 —— #
        if hl_cross:
            hlw_map = {e: float(w) for e, w in zip(hl_cross, hlw_cross)}
            straight = [e for e in hl_cross if (e[1], e[0]) not in has_rev]
            curved   = [e for e in hl_cross if (e[1], e[0]) in  has_rev]
            _draw_dir_edges(
                straight, cross_highlight_color, highlight_alpha,
                zorder=3, curved=False,
                widths=[hlw_map[e] for e in straight]
            )
            _draw_dir_edges(
                curved, cross_highlight_color, highlight_alpha,
                zorder=3, curved=True,
                widths=[hlw_map[e] for e in curved]
            )

    # 7) —— 节点放在最后渲染（提 zorder，避免颜色污染）——
    nodes_artist = nx.draw_networkx_nodes(
        G, pos, nodelist=nodes,
        node_color=[color_map.get(node_to_label[n], unassigned_color) for n in nodes],
        node_size=node_sizes, edgecolors='gray', linewidths=0.5, alpha=0.96, ax=ax
    )
    _safe_set_zorder(nodes_artist, 5)

    # 标签也放最后一层之上
    if show_labels:
        label_artists = nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in nodes}, font_size=8, ax=ax)
        try:
            for t in label_artists.values():
                t.set_zorder(6)
        except Exception:
            pass

    # ----------- 图例（底部多列，编号从1开始）-----------
    if order_labels:
        disp_map = {lbl: i+1 for i, lbl in enumerate(order_labels)}
        handles = [Patch(facecolor=color_map[lbl], label=f'Community {disp_map[lbl]}')
                   for lbl in order_labels]
        K = len(order_labels)
        ncol = 2 if K<=10 else 3 if K<=20 else 4 if K<=30 else 5
        ax.legend(handles=handles,
                  loc='upper center', bbox_to_anchor=(0.5, -0.06),
                  ncol=ncol, fontsize=9, frameon=True,
                  columnspacing=1.2, handlelength=1.4, borderaxespad=0.2)

    ax.set_axis_off()
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()






def selected_labels(names: list[str] | None) -> set[str] | None:
    if not names:
        return None
    valid = {name.lower() for name in MODEL_ORDER}
    requested = {name.lower() for name in names}
    unknown = sorted(requested - valid)
    if unknown:
        raise ValueError(f"Unknown model name(s): {', '.join(unknown)}")
    return requested


def build_graph_inputs(name: str, weight: np.ndarray, seed: int, vanilla_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.abs(weight.copy())
    np.fill_diagonal(matrix, 0)
    adjacency = binarize_directed(remove_isolated_nodes_like(matrix))
    np.fill_diagonal(adjacency, 0)

    if name.lower() == "vanilla":
        sample_size = min(int(vanilla_nodes), adjacency.shape[0])
        rng = np.random.default_rng(seed)
        sample_idx = np.sort(rng.choice(adjacency.shape[0], size=sample_size, replace=False))
        adjacency = adjacency[np.ix_(sample_idx, sample_idx)]
        labels = np.zeros(sample_size, dtype=int)
        print(f"Vanilla example nodes: {sample_idx.tolist()}")
        return adjacency, labels

    labels_best, _score_best, _scores = run_infomap_repeats(
        adjacency,
        n_runs=2,
        seed_base=seed,
        trials=50,
        two_level=True,
    )
    labels = np.asarray(labels_best, dtype=int).reshape(-1)
    return adjacency, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--models", nargs="+", help="Subset to render, e.g. FRP MOp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vanilla-nodes", type=int, default=25)
    args = parser.parse_args()

    wanted = selected_labels(args.models)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_map = {label: weight for label, weight in load_models(args.model_dir)}

    for label in MODEL_ORDER:
        if wanted is not None and label.lower() not in wanted:
            continue
        if label not in model_map:
            raise FileNotFoundError(f"Missing model checkpoint for {label} in {args.model_dir}")
        adjacency, labels = build_graph_inputs(label, model_map[label], args.seed, args.vanilla_nodes)
        out_path = args.output_dir / f"network_{label}.png"
        plot_network(adjacency, labels=labels, save_path=str(out_path))
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
