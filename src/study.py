"""最小台数の探索・1か月評価・年間費用（バッチ計算と画面で共通）。"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .engine import EB, KB, Fleet, Layout, Ops, allocate_fleet, month_calendar, peak_outflow, simulate_days, simulate_month
from .plan import fleet_for, make_ops


# ---------------------------------------------------------------- 最小台数
def min_fleet(L: Layout, ops: Ops, out_h, ret_h, targets=(0.05, 0.01, 0.20), n_rep=60, seed=0,
              max_each: int | None = None) -> dict:
    """自転車台数×キックボード台数の組み合わせのうち、貸出失敗率 ≤ 目標 を満たす最小の合計台数。
    失敗率は各車種の台数に対して減少するとみなし、自転車台数ごとに必要なキックボード台数の下限を
    階段状にたどる（総当たりと同じ答えを、少ない評価回数で求める）。同じ乱数列で比べる。
    戻り値：{目標: {"自転車", "キックボード", "合計", "失敗率"} or None}, "_cache": {(nb,nk): 失敗率}"""
    peak = peak_outflow(out_h, ret_h) * sum(m * d for m, d in zip(ops.mix, ops.demand_mult))
    if max_each is None:
        max_each = int(math.ceil(peak * 2.5)) + 8
    cache: dict[tuple, float] = {}

    def f(nb, nk):
        if (nb, nk) not in cache:
            F = allocate_fleet(L, ops, nb, nk, out_h, ret_h, seed=seed)
            cache[(nb, nk)] = simulate_days(L, F, ops, out_h, ret_h, n_rep=n_rep, seed=seed)["貸出失敗率"]
        return cache[(nb, nk)]

    out = {}
    for tgt in sorted(targets, reverse=True):
        best, nk = None, max_each
        for nb in range(0, max_each + 1):
            if best is not None and nb >= best[2]:
                break
            if f(nb, max_each) > tgt:
                continue
            nk = min(nk, max_each)
            while nk > 0 and f(nb, nk - 1) <= tgt:
                nk -= 1
            tot = nb + nk
            ratio_gap = abs(nb / tot - ops.mix[EB]) if tot else 1
            if best is None or tot < best[2] or (tot == best[2] and ratio_gap < best[3]):
                best = (nb, nk, tot, ratio_gap)
        out[tgt] = None if best is None else {"自転車": best[0], "キックボード": best[1], "合計": best[2],
                                              "失敗率": f(best[0], best[1])}
    out["_cache"] = cache
    return out


# ---------------------------------------------------------------- 1か月
def month_eval(L: Layout, fleet: Fleet, cfg, demand_by_daytype: dict, *, pattern="a", substitution=None,
               ebike_share=None, procedure_sec=None, walk_kmh=None, month=None, seed=0):
    """demand_by_daytype: {"平日": (out_h, ret_h), "休日": (...)}（気象補正前）。気象は月の平年出現日数から日ごとに抽選。"""
    mcfg = cfg["month"]
    m = month or mcfg["month"]
    rng = np.random.default_rng(seed + 7919)
    cal = month_calendar(mcfg["year"], m, mcfg.get(f"holidays_{mcfg['year']}", []), cfg["climate"][m], rng)
    ops_cache = {}

    def ops_for(w):
        if w not in ops_cache:
            ops_cache[w] = make_ops(cfg, weather=w, pattern=pattern, substitution=substitution,
                                    ebike_share=ebike_share, procedure_sec=procedure_sec, walk_kmh=walk_kmh)
        return ops_cache[w]

    summary, daily, days = simulate_month(L, fleet, ops_for, lambda dt, w: demand_by_daytype[dt], cal, seed=seed)
    return summary, daily


# ---------------------------------------------------------------- 費用
def annual_cost(cfg, fleet: Fleet, k_house: int, daily: pd.DataFrame, station_cap: int,
                price_scale=1.0, port_scale=1.0, truck_scale=1.0) -> dict:
    """年間費用（円）。平日・休日それぞれの1日平均（1か月シミュレーション）を稼働日数で年換算する。"""
    c = cfg["cost"]
    nb, nk = fleet.n
    n = nb + nk
    vehicle = (nb * c["price"]["ebike"] + nk * c["price"]["kickboard"]) * price_scale / c["life_years"]
    maintenance = n * c["maintenance"]
    ports = (k_house + 2) * c["port_initial"] * port_scale / c["port_life_years"]
    # 区画数：街区側＝初期台数と最大駐車の大きい方、駅＝容量、モビリティステーション＝最大駐車
    peaks = np.array(daily["ポート別最大駐車(95%点)"].tolist()).max(0) if len(daily) else np.zeros(k_house + 2)
    house_slots = np.maximum(fleet.init[:k_house].sum(1), peaks[:k_house]).sum()
    slots = int(house_slots + station_cap + max(peaks[k_house + 1], 1))
    land = slots * c["land_per_slot_month"] * 12
    by = {dt: daily[daily["曜日区分"] == dt] for dt in ("平日", "休日")}
    days = c["days"]
    op_min = sum(days[dt] * by[dt]["作業時間(分/日)"].mean() for dt in days if len(by[dt]))
    op_n = sum(days[dt] * by[dt]["作業回数/日"].mean() for dt in days if len(by[dt]))
    labor = op_min / 60 * c["labor_per_hour"]
    truck = op_n * c["truck_per_op"] * truck_scale
    trips = sum(days[dt] * by[dt]["利用"].mean() for dt in days if len(by[dt]))
    total = vehicle + maintenance + ports + land + labor + truck
    rebalance = labor + truck
    return {"車両（償却）": vehicle, "車両維持": maintenance, "ポート（償却）": ports, "用地": land,
            "人件費（リバランス）": labor, "軽トラック": truck, "年間総費用": total,
            "区画数": slots, "年間利用回数": trips,
            "1トリップあたり費用": total / trips if trips > 0 else float("nan"),
            "1台あたり年間費用": total / n if n else float("nan"),
            "再配置費用の割合": rebalance / total if total else 0.0,
            "運営費に占める再配置費用": rebalance / (maintenance + land + labor + truck) if (maintenance + land + labor + truck) else 0.0,
            "稼働率(回/台/日)": trips / sum(days.values()) / n if n else float("nan")}
