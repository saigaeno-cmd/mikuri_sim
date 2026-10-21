"""最小台数の探索・1か月評価・年間費用（バッチ計算と画面で共通）。"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .engine import (EB, KB, Fleet, Layout, Ops, allocate_fleet, month_calendar, peak_outflow, simulate_day, summarize,
                     simulate_days, simulate_month)
from .plan import fleet_for, make_ops


# ---------------------------------------------------------------- 最小台数
class _SeqEval:
    """（自転車, キックボード）ごとの貸出失敗率を、日を追加しながら評価する（逐次判定）。
    日 i の乱数は台数によらず同じ（共通乱数）。目標との大小が統計的にはっきりした時点で打ち切り、
    はっきりしなければ n_rep 日まで計算する。"""

    def __init__(self, L, ops, out_h, ret_h, n_rep, seed, chunk=10, z=2.5):
        self.a = (L, ops, out_h, ret_h)
        self.n_rep, self.seed, self.chunk, self.z = n_rep, seed, chunk, z
        self.st: dict[tuple, list] = {}        # key -> [fails(list), reqs(list)]
        self.days_run = 0

    def _extend(self, key, upto):
        L, ops, out_h, ret_h = self.a
        if key not in self.st:
            self.st[key] = [[], [], allocate_fleet(L, ops, key[0], key[1], out_h, ret_h, seed=self.seed)]
        f, q, F = self.st[key]
        for i in range(len(f), min(upto, self.n_rep)):
            d, _ = simulate_day(L, F, ops, out_h, ret_h, np.random.default_rng([self.seed, i]))
            f.append(d.fail_out + d.fail_ret)
            q.append(d.req_out + d.req_ret)
            self.days_run += 1

    def rate(self, key, full=False):
        if full:
            self._extend(key, self.n_rep)
        f, q, _ = self.st[key]
        return sum(f) / sum(q) if sum(q) else 0.0

    def le(self, key, tgt) -> bool:
        """失敗率 ≤ tgt か。"""
        while True:
            n = len(self.st[key][0]) if key in self.st else 0
            if n >= self.n_rep:
                return self.rate(key) <= tgt
            if n >= self.chunk:
                f, q, _ = self.st[key]
                f, q = np.asarray(f, float), np.asarray(q, float)
                if q.sum() > 0:
                    r = f.sum() / q.sum()
                    # 日ごとのばらつきから求めた比率推定量の標準誤差（日内の相関を含む）
                    se = np.sqrt(max(np.var(f - r * q, ddof=1), 1e-9) / n) / max(q.mean(), 1e-9)
                    if r + self.z * se < tgt:
                        return True
                    if r - self.z * se > tgt:
                        return False
            self._extend(key, n + self.chunk)


def min_fleet(L: Layout, ops: Ops, out_h, ret_h, targets=(0.05, 0.01, 0.20), n_rep=60, seed=0,
              max_each: int | None = None) -> dict:
    """自転車台数×キックボード台数の組み合わせのうち、貸出失敗率 ≤ 目標 を満たす最小の合計台数。
    失敗率は各車種の台数に対して減少するとみなし、自転車台数ごとに必要なキックボード台数の下限を
    階段状にたどる（総当たりと同じ答えを、少ない評価回数で求める）。同じ乱数列で比べ、
    目標との大小がはっきりした組み合わせは途中の日数で打ち切る（逐次判定）。
    戻り値：{目標: {"自転車", "キックボード", "合計", "失敗率"} or None}, "_cache": {(nb,nk): 計算日数}, "_days": 総計算日数"""
    peak = peak_outflow(out_h, ret_h) * sum(m * d for m, d in zip(ops.mix, ops.demand_mult))
    if max_each is None:
        max_each = int(math.ceil(peak * 2.5)) + 8
    ev = _SeqEval(L, ops, out_h, ret_h, n_rep, seed)

    out = {}
    for tgt in sorted(targets, reverse=True):
        best, nk = None, max_each
        for nb in range(0, max_each + 1):
            if best is not None and nb >= best[2]:
                break
            if not ev.le((nb, max_each), tgt):
                continue
            nk = min(nk, max_each)
            while nk > 0 and ev.le((nb, nk - 1), tgt):
                nk -= 1
            tot = nb + nk
            ratio_gap = abs(nb / tot - ops.mix[EB]) if tot else 1
            if best is None or tot < best[2] or (tot == best[2] and ratio_gap < best[3]):
                best = (nb, nk, tot, ratio_gap)
        out[tgt] = None if best is None else {"自転車": best[0], "キックボード": best[1], "合計": best[2],
                                              "失敗率": ev.rate((best[0], best[1]), full=True)}
    out["_cache"] = {k: len(v[0]) for k, v in ev.st.items()}
    out["_days"] = ev.days_run
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

    summary, daily, days = simulate_month(L, fleet, ops_for, lambda dt, w: demand_by_daytype[dt], cal, seed=seed,
                                          discourage=_discourage(cfg))
    return summary, daily


def _discourage(cfg) -> float:
    return float((cfg.get("behavior") or {}).get("discourage_alpha") or 0.0)


def year_eval(L: Layout, fleet: Fleet, cfg, demand_by_daytype: dict, *, pattern="a", substitution=None,
              ebike_share=None, procedure_sec=None, walk_kmh=None, seed=0):
    """1年（12か月、気象は月ごとの平年出現日数から抽選）を連続で計算する。
    戻り値：(年間の集計, 日別の表, 月別の表)。年間費用はこの日別の表から計算する。"""
    mcfg = cfg["month"]
    year = mcfg["year"]
    rng = np.random.default_rng(seed + 7919)
    hol = mcfg.get(f"holidays_{year}", [])
    cal = pd.concat([month_calendar(year, m, hol, cfg["climate"][m], rng) for m in range(1, 13)], ignore_index=True)
    ops_cache = {}

    def ops_for(w):
        if w not in ops_cache:
            ops_cache[w] = make_ops(cfg, weather=w, pattern=pattern, substitution=substitution,
                                    ebike_share=ebike_share, procedure_sec=procedure_sec, walk_kmh=walk_kmh)
        return ops_cache[w]

    summary, daily, days = simulate_month(L, fleet, ops_for, lambda dt, w: demand_by_daytype[dt], cal, seed=seed,
                                          discourage=_discourage(cfg))
    daily["月"] = pd.to_datetime(daily["日付"]).dt.month
    by_month = []
    for m, idx in daily.groupby("月").groups.items():
        sm = summarize([days[i] for i in idx], fleet)
        by_month.append({"月": m, **{k: v for k, v in sm.items() if not isinstance(v, list)}})
    return summary, daily, pd.DataFrame(by_month)


# ---------------------------------------------------------------- 費用
def annual_cost(cfg, fleet: Fleet, k_house: int, daily: pd.DataFrame, station_cap: int,
                price_scale=1.0, port_scale=1.0, truck_scale=1.0) -> dict:
    """年間費用（円）。daily が1年分（year_eval）ならその合計、1か月分なら平日・休日の1日平均を稼働日数で年換算する。"""
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
    if len(daily) >= 360:                      # 1年分の日別結果：そのまま合計（365日に換算）
        sc = 365 / len(daily)
        op_min = daily["作業時間(分/日)"].sum() * sc
        op_n = daily["作業回数/日"].sum() * sc
        trips = daily["利用"].sum() * sc
        swaps = daily["電池交換/日"].sum() * sc
        n_days = 365
    else:                                      # 1か月分：平日・休日の1日平均を稼働日数で年換算
        by = {dt: daily[daily["曜日区分"] == dt] for dt in ("平日", "休日")}
        days = c["days"]
        op_min = sum(days[dt] * by[dt]["作業時間(分/日)"].mean() for dt in days if len(by[dt]))
        op_n = sum(days[dt] * by[dt]["作業回数/日"].mean() for dt in days if len(by[dt]))
        trips = sum(days[dt] * by[dt]["利用"].mean() for dt in days if len(by[dt]))
        swaps = sum(days[dt] * by[dt]["電池交換/日"].mean() for dt in days if len(by[dt]))
        n_days = sum(days.values())
    swap_min = (cfg.get("battery", {}).get("kickboard", {}) or {}).get("swap_work_min", 0) or 0
    labor = op_min / 60 * c["labor_per_hour"]
    swap_labor = swaps * swap_min / 60 * c["labor_per_hour"]
    truck = op_n * c["truck_per_op"] * truck_scale
    total = vehicle + maintenance + ports + land + labor + truck + swap_labor
    rebalance = labor + truck
    opex = maintenance + land + labor + truck + swap_labor
    return {"車両（償却）": vehicle, "車両維持": maintenance, "ポート（償却）": ports, "用地": land,
            "人件費（リバランス）": labor, "軽トラック": truck, "人件費（電池交換）": swap_labor, "年間総費用": total,
            "区画数": slots, "年間利用回数": trips,
            "1トリップあたり費用": total / trips if trips > 0 else float("nan"),
            "1台あたり年間費用": total / n if n else float("nan"),
            "再配置費用の割合": rebalance / total if total else 0.0,
            "運営費に占める再配置費用": rebalance / opex if opex else 0.0,
            "稼働率(回/台/日)": trips / n_days / n if n else float("nan")}
