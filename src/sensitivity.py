"""1シナリオの計算と、パラメータ掃引。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .demand import DESTS, build_demand, dest_shares, hourly_matrix
from .fleet import FleetResult, simulate_fleet
from .loaders import Network
from .location import Solution, cost_matrix, solve_pmedian


@dataclass
class Scenario:
    demand: pd.DataFrame
    shares: np.ndarray
    solution: Solution
    fleet: FleetResult
    walk_min: float       # 平均 徒歩（自宅→ポート）
    ride_min: float       # 平均 走行
    egress_min: float     # 平均 徒歩（目的地ポート→施設）

    @property
    def total_min(self) -> float:
        return self.walk_min + self.ride_min + self.egress_min


def _ordered_E_R(net: Network):
    idx = [net.dest_keys.index(d) for d in DESTS]
    return net.R[:, idx], net.E[idx]


def solve_location(net: Network, shares: np.ndarray, walk_kmh: float, ride_kmh: float, k: int):
    R, E = _ordered_E_R(net)
    C, walk, ride, egress = cost_matrix(net.W, R, E, shares, walk_kmh, ride_kmh)
    sol = solve_pmedian(C, k)
    chosen = sol.ports[sol.assign]
    rows = np.arange(len(chosen))
    return sol, float(walk[rows, chosen].mean()), float(ride[chosen].mean()), egress


def run_scenario(net: Network, survey: pd.DataFrame, cfg: dict, *, conversion: float,
                 purposes: list[str] | None, station_share: float | None,
                 vehicle: str, k: int, opts: dict | None = None) -> Scenario:
    demand = build_demand(survey, cfg, conversion, purposes, station_share, **(opts or {}))
    shares = dest_shares(demand)
    sol, w, r, e = solve_location(net, shares, cfg["walk_speed_kmh"],
                                  cfg["vehicles"][vehicle]["speed_kmh"], k)
    port_share = np.bincount(sol.assign, minlength=len(sol.ports)) / len(sol.assign)
    fleet = simulate_fleet(hourly_matrix(demand, "outbound"), hourly_matrix(demand, "return"),
                           port_share, cfg["simulation"]["service_level"])
    return Scenario(demand, shares, sol, fleet, w, r, e)


def sweep_fleet(net, survey, cfg, *, purposes, station_share, vehicles, ks, conversions, opts=None) -> pd.DataFrame:
    """必要台数：転換率 × ポート数 × 車種。"""
    rows = []
    for v in vehicles:
        for k in ks:
            for c in conversions:
                s = run_scenario(net, survey, cfg, conversion=c, purposes=purposes,
                                 station_share=station_share, vehicle=v, k=k, opts=opts)
                t = s.fleet.table
                rows.append({
                    "車種": cfg["vehicles"][v]["label"], "ポート数": k, "転換率": c,
                    "必要台数": s.fleet.total,
                    "街区側": int(t.loc[t["種別"] == "街区", "初期台数"].sum()),
                    "駅側": int(t.loc[t["種別"] == "station", "初期台数"].sum()),
                    "イオン側": int(t.loc[t["種別"] == "aeon", "初期台数"].sum()),
                    "駅ポート区画": s.fleet.station_slots,
                    "街区側区画": int(t.loc[t["種別"] == "街区", "必要区画"].sum()),
                    "イオン区画": int(t.loc[t["種別"] == "aeon", "必要区画"].sum()),
                })
    return pd.DataFrame(rows)


def sweep_location(net, survey, cfg, *, purposes, vehicles, ks, station_shares, opts=None) -> pd.DataFrame:
    """平均所要時間と最適ポート：ポート数 × 駅の割合 × 車種。（位置は転換率に依存しない）"""
    rows = []
    for v in vehicles:
        for sh in station_shares:
            demand = build_demand(survey, cfg, 1.0, purposes, sh, **(opts or {}))
            shares = dest_shares(demand)
            for k in ks:
                sol, w, r, e = solve_location(net, shares, cfg["walk_speed_kmh"],
                                              cfg["vehicles"][v]["speed_kmh"], k)
                rows.append({
                    "車種": cfg["vehicles"][v]["label"], "vehicle": v, "ポート数": k,
                    "駅の割合": sh, "徒歩(分)": w, "走行(分)": r, "目的地側徒歩(分)": e,
                    "合計(分)": w + r + e, "ports": tuple(int(p) for p in sol.ports),
                })
    return pd.DataFrame(rows)


def station_slots(survey, cfg, *, conversion, purposes, station_share, opts=None) -> int:
    """駅ポートの必要区画。駅ポートの出入りは街区側の配置・車種に依存しない。"""
    demand = build_demand(survey, cfg, conversion, purposes, station_share, **(opts or {}))
    return simulate_fleet(hourly_matrix(demand, "outbound"), hourly_matrix(demand, "return"),
                          np.array([1.0]), cfg["simulation"]["service_level"]).station_slots


def max_conversion_within(survey, cfg, cap: int, *, purposes, station_share, opts=None, tol: float = 0.005) -> float:
    """駅ポートの必要区画が cap 以下に収まる最大の転換率（二分探索）。"""
    def ok(c):
        return station_slots(survey, cfg, conversion=c, purposes=purposes, station_share=station_share,
                             opts=opts) <= cap
    if ok(1.0):
        return 1.0
    if not ok(tol):
        return 0.0
    lo, hi = tol, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo
