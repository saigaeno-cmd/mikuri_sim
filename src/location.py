"""ポート最適配置：需要加重の（徒歩時間＋走行時間）合計を最小化する p-median。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def kmh_to_m_per_min(v: float) -> float:
    return v * 1000 / 60


def cost_matrix(W, R, E, shares, walk_kmh: float, ride_kmh: float):
    """住宅i がポートj を使うときの1トリップあたり所要時間（分）。

    c[i,j] = 徒歩(i→j) + Σ_d 比率_d × (走行(j→d) + 徒歩(dポート→施設))
    戻り値: (総コスト, 徒歩成分, 走行成分(列ベクトル), 目的地側徒歩成分(スカラー))
    """
    vw, vr = kmh_to_m_per_min(walk_kmh), kmh_to_m_per_min(ride_kmh)
    walk = W / vw
    ride = (R / vr) @ shares
    egress = float(E / vw @ shares)
    return walk + ride[None, :] + egress, walk, ride, egress


@dataclass
class Solution:
    ports: np.ndarray    # 選ばれた候補 index（k個）
    assign: np.ndarray   # 住宅ごとの割当先（ports 内の位置 0..k-1）
    objective: float     # Σ 重み × コスト


def _evaluate(C, w, S):
    sub = C[:, S]
    return float(w @ sub.min(1))


def solve_pmedian(C: np.ndarray, k: int, weights: np.ndarray | None = None, max_iter: int = 100) -> Solution:
    """貪欲法で初期解 → Teitz-Bart 入れ替え法で改善（ベクトル化）。"""
    n, m = C.shape
    k = min(k, m)
    w = np.ones(n) if weights is None else np.asarray(weights, float)

    # 貪欲法
    S: list[int] = []
    best = np.full(n, np.inf)
    for _ in range(k):
        totals = w @ np.minimum(best[:, None], C)
        totals[S] = np.inf
        j = int(np.argmin(totals))
        S.append(j)
        best = np.minimum(best, C[:, j])

    # 入れ替え法：位置 s を候補 c に替えて改善するかを全組で評価し、最良の入れ替えを採用
    S_arr = np.array(S)
    obj = _evaluate(C, w, S_arr)
    for _ in range(max_iter):
        best_gain, best_move = 1e-9, None
        for pos in range(k):
            others = np.delete(S_arr, pos)
            base = C[:, others].min(1) if len(others) else np.full(n, np.inf)
            totals = w @ np.minimum(base[:, None], C)
            totals[S_arr] = np.inf
            c = int(np.argmin(totals))
            gain = obj - totals[c]
            if gain > best_gain:
                best_gain, best_move = gain, (pos, c)
        if best_move is None:
            break
        S_arr[best_move[0]] = best_move[1]
        obj = _evaluate(C, w, S_arr)

    S_arr = np.sort(S_arr)
    assign = C[:, S_arr].argmin(1)
    return Solution(ports=S_arr, assign=assign, objective=_evaluate(C, w, S_arr))
