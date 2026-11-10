"""時間短縮にもとづくポート最適化。

需要が発生する条件（毎回の計算で A・X を指定）：
- 住宅（出発地点）からポートまでの徒歩が A 分以内
- 「徒歩だけで行く」より「徒歩＋モビリティ（手続き・目的地側の徒歩を含む）」の合計移動時間が X 分以上短くなる
両方を満たす住宅は、その目的地へ必ず利用する（需要の量は転換率ではなく、この条件だけで決める）。
- ポート位置：X 分以上短くなる住宅（目的地の割合で重み付け）が最も多くなる k か所（最大カバー問題、貪欲法＋入れ替え）
- 需要：条件を満たす住宅が 1日 trips 往復（目的地は割合で按分）。時間帯の分布はアンケートの形を使う
- 台数：この需要で、選んだ1か月の暦（平日・休日・天気）で目標を満たす最小台数（他のページと同じ探索）
"""
from __future__ import annotations

import numpy as np

from .demand import DESTS
from .engine import Layout
from .plan import demand_hourly, dest_caps, mix_ride_kmh, walk_alt_matrix
from .location import kmh_to_m_per_min


def time_tensors(net, cfg, ebike_share: float, *, walk_kmh=None, procedure_sec=None):
    """(徒歩だけの時間 (h,2), モビリティを使う時間 (h,候補,2)) 分。"""
    walk_kmh = walk_kmh or cfg["walk_speed_kmh"]
    proc = (cfg["procedure_sec"] if procedure_sec is None else procedure_sec) / 60
    vw = kmh_to_m_per_min(walk_kmh)
    vr = kmh_to_m_per_min(mix_ride_kmh(cfg, ebike_share))
    idx = [net.dest_keys.index(d) for d in DESTS]
    walk_only = walk_alt_matrix(net, cfg) / vw                                  # (h, 2)
    mob = (net.W / vw)[:, :, None] + proc + (net.R[:, idx] / vr)[None, :, :] + (net.E[idx] / vw)[None, None, :]
    return walk_only, mob


def access_time(net, cfg, *, walk_kmh=None) -> np.ndarray:
    """(h, 候補) 住宅→ポート候補の徒歩（分）。"""
    return net.W / kmh_to_m_per_min(walk_kmh or cfg["walk_speed_kmh"])


def limit_access(mob, access, max_access_min):
    """ポートまでの徒歩が max_access_min 分を超える（住宅, ポート）は使えない（所要時間を ∞ にする）。"""
    if max_access_min is None:
        return mob
    return np.where((access <= max_access_min + 1e-9)[:, :, None], mob, np.inf)


def coverage(walk_only, mob, S, threshold, share):
    """ポート集合 S のとき、各住宅・目的地が X 分以上短くなるか (h,2) と、最良の短縮時間 (h,2)。"""
    best = mob[:, list(S), :].min(1)
    saving = walk_only - best
    return saving >= threshold, saving


def objective(walk_only, mob, S, threshold, share):
    q, saving = coverage(walk_only, mob, S, threshold, share)
    return float((q * share).sum()), float((np.clip(saving, 0, None) * q * share).sum())


def solve_max_cover(walk_only, mob, k: int, threshold: float, share, max_iter: int = 50) -> list[int]:
    """X 分以上短くなる住宅（目的地の割合で重み付け）が最大になる k か所。同点なら短縮時間の合計が大きい方。"""
    n_cand = mob.shape[1]
    S: list[int] = []
    for _ in range(min(k, n_cand)):
        best = max((objective(walk_only, mob, S + [j], threshold, share), j) for j in range(n_cand) if j not in S)
        S.append(best[1])
    cur = objective(walk_only, mob, S, threshold, share)
    for _ in range(max_iter):                     # 1つずつ入れ替えて良くなる限り続ける
        improved = False
        for a in range(len(S)):
            for j in range(n_cand):
                if j in S:
                    continue
                T = S[:a] + [j] + S[a + 1:]
                v = objective(walk_only, mob, T, threshold, share)
                if v > cur:
                    S, cur, improved = T, v, True
        if not improved:
            break
    return S


def layout_for(net, cfg, S, q, share, walk_only, mob, *, station_cap=None) -> tuple[Layout, np.ndarray, np.ndarray]:
    """選んだポートの配置。住宅は（条件を満たす目的地の）所要時間が最小のポートを使う。
    戻り値：(Layout, 住宅ごとの割当 (使わない住宅は -1), 住宅ごとの重み＝1日あたりの往復数の割合)。"""
    k = len(S)
    w_hd = q * share                                                             # (h,2)
    w = w_hd.sum(1)
    mob = np.where(np.isfinite(mob), mob, 1e6)                                  # 使えないポート（徒歩が遠い）
    cost = (mob[:, S, :] * np.where(w_hd.sum(1, keepdims=True) > 0, w_hd, share)[:, None, :]).sum(2)  # (h,k)
    assign = np.where(w > 0, cost.argmin(1), -1)
    idx = [net.dest_keys.index(d) for d in DESTS]
    Walt = walk_alt_matrix(net, cfg)
    share_k = np.array([w[assign == j].sum() for j in range(k)])
    tot = share_k.sum()
    walk_m, alt_m = np.zeros(k), np.zeros((k, 2))
    for j in range(k):
        m = assign == j
        ww = w[m] if m.any() and w[m].sum() > 0 else None
        cols = net.W[:, S[j]]
        if ww is not None:
            walk_m[j] = np.average(cols[m], weights=ww)
            alt_m[j] = np.average(Walt[m], axis=0, weights=ww)
        else:
            walk_m[j] = cols.mean()
            alt_m[j] = Walt.mean(0)
    dc = dest_caps(cfg, station_cap)
    caps = np.array([np.inf] * k + [np.inf if c is None else float(c) for c in dc])
    L = Layout(k=k, share=share_k / tot if tot > 0 else np.full(k, 1 / k), walk_m=walk_m,
               ride_m=net.R[S][:, idx], egress_m=net.E[idx], alt_m=alt_m, caps=caps,
               labels=[f"P{j + 1}" for j in range(k)])
    return L, assign, w


def demand_for(survey, cfg, trips_per_day: dict, day: str):
    """目的地ごとの1日の往復数から、時間帯別の (往路, 復路) をつくる。分布の形はアンケート（転換率100%）。"""
    P = cfg["survey_xlsx"]["default_modes"]
    oh, rh, _ = demand_hourly(survey, cfg, 1.0, P, day)
    out, ret = np.zeros_like(oh), np.zeros_like(rh)
    for d, key in enumerate(DESTS):
        n = trips_per_day.get(key, 0.0)
        so, sr = oh[:, d].sum(), rh[:, d].sum()
        if so > 0:
            out[:, d] = oh[:, d] / so * n
        if sr > 0:
            ret[:, d] = rh[:, d] / sr * n          # 駅：夕方の帰り（片道）。イオン：帰る時刻の分布として使う
    return out, ret
