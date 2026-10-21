"""配置（Layout）・運用条件（Ops）・需要を設定から組み立てる。画面とバッチ計算で共通。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .demand import DESTS, build_demand, dest_shares, hourly_matrix
from .engine import EB, KB, Layout, Ops, allocate_fleet, peak_outflow
from .loaders import Network
from .location import cost_matrix, solve_pmedian
from .geo import haversine
from .weather import get_weather


def walk_alt_matrix(net: Network, cfg: dict) -> np.ndarray:
    """(n_houses, 2) 自宅→施設入口の徒歩距離（m）。入口が複数なら近い方。事前計算の全ノード行列から引く。"""
    nodes = net.nodes[net.nodes["usable"]]
    cols = []
    for d in DESTS:
        dd = cfg["destinations"][d]
        best = None
        for e in dd.get("entrances") or [dd["entrance"]]:
            dist = haversine(nodes["lat"].to_numpy(), nodes["lon"].to_numpy(), e["lat"], e["lon"])
            i = int(np.argmin(dist))
            col = net.Wn[:, nodes.index[i]] + dist[i]
            best = col if best is None else np.minimum(best, col)
        cols.append(best)
    return np.column_stack(cols)


def _idx(net):
    return [net.dest_keys.index(d) for d in DESTS]


def mix_ride_kmh(cfg, ebike_share: float, weather: str | None = None) -> float:
    """車種比率で所要時間を加重平均した実効走行速度（ポート位置の決定に使う）。"""
    v_eb = cfg["vehicles"]["ebike"]["speed_kmh"]
    v_kb = cfg["vehicles"]["kickboard"]["speed_kmh"]
    if weather:
        v_eb *= get_weather(cfg, weather, "ebike").ride
        v_kb *= get_weather(cfg, weather, "kickboard").ride
    return 1.0 / (ebike_share / v_eb + (1 - ebike_share) / v_kb)


def dest_caps(cfg, station_cap=None) -> list:
    sc = station_cap if station_cap is not None else cfg["station_port"].get("max_slots")
    return [sc, cfg.get("service", {}).get("aeon_max_slots")]


def _layout_from(net, cfg, W, R, E, assign, Walt, house_caps, station_cap, labels):
    k = W.shape[1]
    cnt = np.bincount(assign, minlength=k)
    rows = np.arange(len(assign))
    walk_m = np.bincount(assign, weights=W[rows, assign], minlength=k) / np.maximum(cnt, 1)
    alt_m = np.stack([np.bincount(assign, weights=Walt[:, d], minlength=k) for d in range(2)], 1) / np.maximum(cnt, 1)[:, None]
    dc = dest_caps(cfg, station_cap)
    hc = [np.inf] * k if house_caps is None else [np.inf if (c is None or pd.isna(c)) else float(c) for c in house_caps]
    caps = np.array(hc + [np.inf if c is None else float(c) for c in dc])
    return Layout(k=k, share=cnt / len(assign), walk_m=walk_m, ride_m=R, egress_m=E, alt_m=alt_m, caps=caps,
                  labels=labels or [f"P{j + 1}" for j in range(k)])


def optimal_layout(net: Network, cfg, shares, k: int, ebike_share: float, *, walk_kmh=None, weather=None,
                   station_cap=None, house_caps=None, Walt=None):
    """車種比率で加重した（徒歩＋走行＋目的地側徒歩）が最小になる k か所（p-median）。戻り値 (Layout, 候補index, 割当)。"""
    walk_kmh = walk_kmh or cfg["walk_speed_kmh"]
    if weather:
        walk_kmh *= get_weather(cfg, weather, "ebike").walk
    idx = _idx(net)
    C, *_ = cost_matrix(net.W, net.R[:, idx], net.E[idx], shares, walk_kmh, mix_ride_kmh(cfg, ebike_share, weather))
    sol = solve_pmedian(C, k)
    Walt = walk_alt_matrix(net, cfg) if Walt is None else Walt
    L = _layout_from(net, cfg, net.W[:, sol.ports], net.R[sol.ports][:, idx], net.E[idx], sol.assign, Walt,
                     house_caps, station_cap, None)
    return L, sol.ports, sol.assign


def custom_layout(net: Network, cfg, ports: pd.DataFrame, shares, ebike_share: float, *, walk_kmh=None,
                  station_cap=None, Walt=None):
    """任意の地点（snap_ports 済み：node, snap_m, name, 収納）での配置。住宅は所要時間が最小のポートを使う。"""
    walk_kmh = walk_kmh or cfg["walk_speed_kmh"]
    idx = _idx(net)
    node, off = ports["node"].to_numpy(), ports["snap_m"].to_numpy()
    W = net.Wn[:, node] + off[None, :]
    R = net.Rn[node][:, idx] + off[:, None]
    E = net.E[idx]
    C, *_ = cost_matrix(W, R, E, shares, walk_kmh, mix_ride_kmh(cfg, ebike_share))
    assign = C.argmin(1)
    Walt = walk_alt_matrix(net, cfg) if Walt is None else Walt
    caps = ports["収納"].to_numpy() if "収納" in ports else None
    labels = list(ports["name"]) if "name" in ports else None
    return _layout_from(net, cfg, W, R, E, assign, Walt, caps, station_cap, labels), assign


def make_ops(cfg, *, weather="晴れ", pattern="a", substitution=None, ebike_share=None, procedure_sec=None,
             walk_kmh=None, nightly_charge=None, station_buffer_q="cfg", full_return=None) -> Ops:
    fm = cfg["fleet_mix"]
    es = fm["ebike_share"] if ebike_share is None else ebike_share
    w_eb, w_kb = get_weather(cfg, weather, "ebike"), get_weather(cfg, weather, "kickboard")
    b = cfg["battery"]["kickboard"]
    rb = cfg["rebalance"]
    return Ops(
        walk_kmh=(walk_kmh or cfg["walk_speed_kmh"]) * w_eb.walk,
        ride_kmh=(cfg["vehicles"]["ebike"]["speed_kmh"] * w_eb.ride, cfg["vehicles"]["kickboard"]["speed_kmh"] * w_kb.ride),
        mix=(es, 1 - es), demand_mult=(w_eb.demand, w_kb.demand),
        substitution=fm["substitution"] if substitution is None else substitution,
        procedure_min=(cfg["procedure_sec"] if procedure_sec is None else procedure_sec) / 60,
        trip_type=(cfg["trip_type"]["station"], cfg["trip_type"]["aeon"]),
        pattern=pattern, reb_times=(rb["times"]["b"], rb["times"]["c"]),
        truck_cap=(rb["truck_capacity"]["ebike"], rb["truck_capacity"]["kickboard"]),
        base_min=rb["base_min"], per_vehicle_min=rb["per_vehicle_min"], extra_trip_min=rb["extra_trip_min"],
        kb_range_km=b["range_km"], kb_swap_below=b["swap_below"], kb_out_hours=b["out_hours"],
        nightly_charge=b["nightly_charge"] if nightly_charge is None else nightly_charge,
        station_buffer_q=rb.get("station_buffer_q") if station_buffer_q == "cfg" else station_buffer_q,
        full_return=rb.get("full_return", "outside") if full_return is None else full_return,
        collect_delay_min=rb.get("collect_delay_min", 60),
    )


def demand_hourly(survey, cfg, conversion, purposes, day, hourly=None, station_share=None):
    """気象補正前（転換率のみ）の時間帯別需要 (out_h, ret_h) と目的地比率。気象の倍率は Ops.demand_mult で掛ける。"""
    d = build_demand(survey, cfg, conversion, purposes, station_share, day=day, hourly=hourly)
    return hourly_matrix(d, "outbound"), hourly_matrix(d, "return"), dest_shares(d)


def fleet_for(L: Layout, n_eb: int, n_kb: int, out_h, ret_h, ops: Ops, seed: int = 0):
    """毎晩の初期配置（街区ポート＋駅ポート）。"""
    return allocate_fleet(L, ops, n_eb, n_kb, out_h, ret_h, seed=seed)
