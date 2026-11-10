"""候補地（配置可能台数つき）の中から、合計台数と使うポート数を決めたとき充足率が最大になる組み合わせを探す。"""
from __future__ import annotations

import itertools
import math

import pandas as pd

import numpy as np

from .engine import EB, KB, allocate_fleet, simulate_day, simulate_days, summarize
from .plan import custom_layout


def evaluate_subset(net, cfg, cands: pd.DataFrame, idx, n_total, ebike_share, shares, oh, rh, ops, station_cap,
                    n_rep=60, seed=1, pspec=None, day_specs=None) -> dict:
    """候補 idx を使い、合計 n_total 台（自転車の割合 ebike_share）を配置可能台数の範囲で自動配分したときの結果。
    pspec：利用用途（src.purposes）の条件。与えると公園ポートと用途別の流れを加える（oh/rh は with_flows 済み）。
    day_specs：[(Ops, out_h, ret_h)] の暦。与えると日 i をその i % 日数 番目の条件で計算する。"""
    sub = cands.iloc[list(idx)].reset_index(drop=True)
    L, assign = custom_layout(net, cfg, sub, shares, ebike_share, station_cap=station_cap)
    if pspec is not None:
        from .purposes import attach, ride_from_nodes
        L = attach(net, cfg, L, assign, ride_from_nodes(net, sub), pspec)
    nb = int(round(n_total * ebike_share))
    F = allocate_fleet(L, ops, nb, n_total - nb, oh, rh, seed=seed)
    placed = int(F.init.sum())
    if day_specs:
        days = [simulate_day(L, F, *day_specs[i % len(day_specs)], np.random.default_rng([seed, i]))[0]
                for i in range(n_rep)]
        r = summarize(days, F)
    else:
        r = simulate_days(L, F, ops, oh, rh, n_rep=n_rep, seed=seed)
    return {"使うポート": "・".join(sub["name"].astype(str)), "_idx": tuple(int(i) for i in idx),
            "充足率": r["充足率"], "平均移動時間(分/片道)": r["平均移動時間(分/片道)"], "返却失敗率": r["返却失敗率"],
            "置けた台数": placed, "駅ポート初期": int(F.init[L.k].sum()),
            "内訳": "・".join(f"{n}:自{F.init[j, EB]}キ{F.init[j, KB]}" for j, n in enumerate(sub["name"])),
            "合計移動時間(人・分/日)": r["合計移動時間(人・分/日)"],
            "_init": F.init, "_用途別": r.get("_用途別", {}), "_hourly": r.get("_hourly", []),
            "_peaks": r["ポート別最大駐車(95%点)"], "_assign": assign, "_caps": L.caps}


def best_subsets(net, cfg, cands: pd.DataFrame, m: int, n_total: int, ebike_share, shares, oh, rh, ops,
                 station_cap, n_rep=60, max_combos=300, progress=None, pspec=None,
                 day_specs=None) -> tuple[pd.DataFrame, str]:
    """m か所の組み合わせを比べ、充足率の高い順に並べた表と、探索の方法（全組み合わせ／貪欲法）。"""
    n = len(cands)
    total = math.comb(n, m)
    rows = []
    if total <= max_combos:
        method = f"全{total}通りを比較"
        for i, idx in enumerate(itertools.combinations(range(n), m)):
            rows.append(evaluate_subset(net, cfg, cands, idx, n_total, ebike_share, shares, oh, rh, ops, station_cap, n_rep,
                                        pspec=pspec, day_specs=day_specs))
            if progress:
                progress((i + 1) / total)
    else:                                           # 多すぎるときは1か所ずつ足していく（貪欲法）
        method = f"組み合わせが{total}通りと多いため、1か所ずつ最良を足す方法で探索"
        chosen: list[int] = []
        for step in range(m):
            cand_rows = [evaluate_subset(net, cfg, cands, chosen + [j], n_total, ebike_share, shares, oh, rh, ops,
                                         station_cap, n_rep, pspec=pspec, day_specs=day_specs)
                         for j in range(n) if j not in chosen]
            best = max(cand_rows, key=lambda r: (r["充足率"], -r["平均移動時間(分/片道)"]))
            chosen = list(best["_idx"])
            rows.extend(cand_rows if step == m - 1 else [])
            if progress:
                progress((step + 1) / m)
    df = pd.DataFrame(rows).sort_values(["充足率", "平均移動時間(分/片道)"], ascending=[False, True]).reset_index(drop=True)
    return df, method
