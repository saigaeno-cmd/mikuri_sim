"""台数を固定した実証（例：10台）の評価：ポート数・自転車とキックボードの割合ごとの充足率など。"""
from __future__ import annotations

import pandas as pd

from .engine import simulate_days
from .plan import demand_hourly, fleet_for, make_ops, optimal_layout
from .study import annual_cost, year_eval

KEYS = ["充足率", "貸出失敗率", "利用希望", "利用", "返却失敗率", "代替乗車率", "稼働率(回/台/日)", "駅ポート最大在庫",
        "平均移動時間(分/片道)", "全員徒歩の平均(分/片道)"]


def pilot_grid(net, cfg, survey, conversion: float, n_total: int = 10, ks=(1, 2, 3), station_cap=None,
               pattern="a", substitution=None, n_rep=60, seed=0, day="平日") -> pd.DataFrame:
    """住宅側ポート数 k × 自転車台数（残りはキックボード）ごとの1日（晴れ）× n_rep 日の結果。"""
    P = cfg["survey_xlsx"]["default_modes"]
    oh, rh, sh = demand_hourly(survey, cfg, conversion, P, day)
    cap = station_cap or cfg["station_port"]["max_slots"]
    ops = make_ops(cfg, pattern=pattern, substitution=substitution)
    rows = []
    for k in ks:
        L, ports, assign = optimal_layout(net, cfg, sh, k, cfg["fleet_mix"]["ebike_share"], station_cap=cap)
        for nb in range(n_total + 1):
            F = fleet_for(L, nb, n_total - nb, oh, rh, ops, seed=seed)
            r = simulate_days(L, F, ops, oh, rh, n_rep=n_rep, seed=seed)
            rows.append({"転換率": conversion, "ポート数": k, "自転車": nb, "キックボード": n_total - nb,
                         "駅初期台数": int(F.init[L.k].sum()), **{c: r[c] for c in KEYS},
                         "断られた/日": r["利用希望"] - r["利用"]})
    return pd.DataFrame(rows)


def pilot_best(grid: pd.DataFrame) -> pd.Series:
    """充足率が最大の組み合わせ（差が0.5ポイント以内ならポート数が少ない方、希望の車種比率に近い方、移動時間が短い方）。"""
    top = grid["充足率"].max()
    near = grid[grid["充足率"] >= top - 0.005].copy()
    n = near["自転車"] + near["キックボード"]
    near["_gap"] = (near["自転車"] / n - 0.603).abs()                 # 希望の車種比率に近いもの
    return near.sort_values(["ポート数", "_gap", "平均移動時間(分/片道)"]).iloc[0].drop("_gap")


def pilot_year(net, cfg, survey, conversion: float, best: pd.Series, station_cap=None, pattern="a",
               substitution=None, seed=0):
    """選んだ組み合わせで1年（月別・曜日・気象）を計算する。戻り値：(年間の集計, 月別の表, 費用)。"""
    P = cfg["survey_xlsx"]["default_modes"]
    oh, rh, sh = demand_hourly(survey, cfg, conversion, P, "平日")
    oh2, rh2, _ = demand_hourly(survey, cfg, conversion, P, "休日")
    cap = station_cap or cfg["station_port"]["max_slots"]
    L, _, _ = optimal_layout(net, cfg, sh, int(best["ポート数"]), cfg["fleet_mix"]["ebike_share"], station_cap=cap)
    ops = make_ops(cfg, pattern=pattern, substitution=substitution)
    F = fleet_for(L, int(best["自転車"]), int(best["キックボード"]), oh, rh, ops, seed=seed)
    ys, daily, bm = year_eval(L, F, cfg, {"平日": (oh, rh), "休日": (oh2, rh2)}, pattern=pattern,
                              substitution=substitution, seed=seed)
    return ys, bm, annual_cost(cfg, F, L.k, daily, cap)
