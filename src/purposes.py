"""利用用途（2026-11）：住宅⇔駅・住宅⇔イオン（アンケート×転換率）に、次の流れを加える。

3 イオン⇔駅：両方向の片道・毎日（方向ごとに1日の回数）
4 公園⇔住宅：往復・休日のみ（住宅側ポートから公園ポートへ。滞在後に同じ車種で戻る）
5 公園⇔駅  ：往復・休日のみ（駅に電車で来た人が駅ポートから公園へ。滞在後に駅へ戻る）

需要は時間帯の範囲 [開始, 終了) に一様に発生する期待値（件/日）。気象の倍率は Ops.demand_mult で掛かる。
エンジンには out_h の 3列目以降（列 2+i が FLOWS[i]）として渡す。
"""
from __future__ import annotations

import copy
import dataclasses

import numpy as np

from .engine import Flow, Layout
from .geo import haversine

PARK = "park"
# (表示名, 設定キー, 出発, 到着, 往復)。出発・到着："house"=住宅側、それ以外は目的地キー
FLOWS = [
    ("イオン⇔駅", ("aeon_station", "aeon_to_station"), "aeon", "station", False),
    ("イオン⇔駅", ("aeon_station", "station_to_aeon"), "station", "aeon", False),
    ("公園⇔住宅", ("park_house", None), "house", PARK, True),
    ("公園⇔駅", ("park_station", None), "station", PARK, True),
]
HOLIDAY_ONLY = {"park_house", "park_station"}


def default_spec(cfg) -> dict:
    return copy.deepcopy(cfg.get("purposes") or {})


def has_park(net) -> bool:
    return PARK in net.dest_keys and net.DR is not None and net.DW is not None


def walk_to(net, cfg, key) -> np.ndarray:
    """(n_houses,) 自宅→施設 key の入口までの徒歩（m）。入口が複数なら近い方。"""
    nodes = net.nodes[net.nodes["usable"]]
    dd = cfg["destinations"][key]
    best = None
    for e in dd.get("entrances") or [dd["port"]]:
        dist = haversine(nodes["lat"].to_numpy(), nodes["lon"].to_numpy(), e["lat"], e["lon"])
        i = int(np.argmin(dist))
        col = net.Wn[:, nodes.index[i]] + dist[i]
        best = col if best is None else np.minimum(best, col)
    return best


def attach(net, cfg, L: Layout, assign: np.ndarray, ride_house_park: np.ndarray, spec: dict, *,
           park_cap=None) -> Layout:
    """配置 L に公園ポート（番号 k+2）と用途別の流れ FLOWS を加えた Layout。
    ride_house_park：(k,) 住宅側ポート→公園ポートの走行距離（m）。assign：住宅ごとの使うポート。"""
    k = L.k
    di = {d: net.dest_keys.index(d) for d in ("station", "aeon", PARK)}
    port = {"station": k, "aeon": k + 1, PARK: k + 2, "house": -1}
    E = net.E
    w_park = walk_to(net, cfg, PARK)
    alt_hp = np.array([w_park[assign == j].mean() if (assign == j).any() else w_park.mean() for j in range(k)])
    one = np.ones(k)
    flows = []
    for label, (grp, _), a, b, rt in FLOWS:
        if a == "house":
            ride, wo, wd, alt = np.asarray(ride_house_park, float), L.walk_m, one * E[di[b]], alt_hp
        else:
            ride = one * net.DR[di[a], di[b]]
            wo, wd, alt = one * E[di[a]], one * E[di[b]], one * net.DW[di[a], di[b]]
        dwell = float((spec.get(grp) or {}).get("dwell_hours", 0) or 0) * 60 if rt else 0.0
        flows.append(Flow(label=label, origin=port[a], dest=port[b], round_trip=rt, dwell_min=dwell,
                          ride_m=ride, walk_o_m=wo, walk_d_m=wd, alt_m=alt))
    cap = (cfg["destinations"][PARK].get("max_slots") if park_cap is None else park_cap)
    caps = np.append(L.caps[:k + 2], np.inf if cap is None else float(cap))
    return dataclasses.replace(L, caps=caps, flows=flows)


def _uniform(n: float, hours) -> np.ndarray:
    h0, h1 = int(hours[0]), int(hours[1])
    v = np.zeros(24)
    if n > 0 and h1 > h0:
        v[h0:h1] = n / (h1 - h0)
    return v


def flow_columns(spec: dict, day: str) -> np.ndarray:
    """(24, len(FLOWS)) 出発の時間帯別期待値（件/日）。公園の用途は休日のみ。"""
    cols = []
    for _, (grp, sub), *_r in FLOWS:
        g = spec.get(grp) or {}
        n = (g.get("per_day") or {}).get(sub, 0.0) if sub else g.get("per_day", 0.0)
        if grp in HOLIDAY_ONLY and day != "休日":
            n = 0.0
        cols.append(_uniform(float(n or 0.0), g.get("hours", (10, 16))))
    return np.column_stack(cols)


def with_flows(oh: np.ndarray, rh: np.ndarray, spec: dict, day: str) -> tuple[np.ndarray, np.ndarray]:
    """アンケート需要 (24,2) に用途別の流れを足した (24, 2+len(FLOWS))。復路側は0（往復は滞在後に自動で戻る）。"""
    fc = flow_columns(spec, day)
    return np.hstack([oh[:, :2], fc]), np.hstack([rh[:, :2], np.zeros_like(fc)])


def ride_from_nodes(net, ports) -> np.ndarray:
    """snap_ports 済みの地点（node, snap_m）→公園ポートの走行距離。"""
    return net.Rn[ports["node"].to_numpy(), net.dest_keys.index(PARK)] + ports["snap_m"].to_numpy()


def per_day_table(spec: dict) -> list[tuple[str, str, float]]:
    """表示用：(用途, 曜日, 片道の回数/日)。"""
    rows = []
    g = spec.get("aeon_station") or {}
    pd_ = g.get("per_day") or {}
    rows.append(("イオン⇔駅", "毎日", float(pd_.get("aeon_to_station", 0)) + float(pd_.get("station_to_aeon", 0))))
    for grp, lab in (("park_house", "公園⇔住宅"), ("park_station", "公園⇔駅")):
        rows.append((lab, "休日", 2 * float((spec.get(grp) or {}).get("per_day", 0))))
    return rows
