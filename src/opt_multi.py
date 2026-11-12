"""①ポート最適化（2026-11 改訂）：目的地（駅・イオン・公園・エニタイム）と用途ごとの使える車種を固定した、時間短縮型の最適化。

固定条件は config の opt_time（夜間1回の再配置・ポート上限5台・手続き30秒・9〜11月の3か月・実測の経路別速度・1日1往復）。
需要：住宅と目的地の組ごとに「ポートまで徒歩 A 分以内」かつ「徒歩だけより X 分以上短い」なら1日1往復（目的地ごと）。
  駅発の用途（駅⇔イオン・駅⇔公園）は回数を固定。
ポート：条件を満たす（住宅, 目的地）の組が最も多くなる住宅側ポート（上限 k か所、最大カバー＋入れ替え）。
目的地側：駅・イオンは既存のポート、公園・エニタイムは入口の近くに固定。駅ポートは上限なしで、最大駐車を「必要容量」とする。
台数：使う車種ごとに、3か月（平日・休日・天気）で目標充足率を満たす最小台数。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import timesave as ts
from .custom_ports import snap_ports
from .demand import DESTS, table_hourly
from .engine import TYPES, Flow, Layout, allocate_fleet, month_calendar, peak_outflow, simulate_day
from .location import kmh_to_m_per_min
from .plan import fleet_for, make_ops, route_kmh, type_speed
from .purposes import walk_to
from .study import annual_cost, min_fleet, month_eval
from .weather_data import month_specs

DEST_ALL = ["station", "aeon", "park", "anytime"]
DEST_LABEL = {"station": "南栗橋駅", "aeon": "イオンスタイル", "park": "公園", "anytime": "エニタイム"}
TYPE_LABEL = {"ebike": "電動アシスト自転車", "kickboard": "電動キックボード", "trans": "4輪モビリティ"}


@dataclass
class Plan:
    """目的地側のポート（番号 k + i）と、用途ごとの設定。"""
    dests: list            # 使う目的地（DEST_ALL の順）
    purposes: list         # 有効な用途（config の辞書に allowed（車種番号）を足したもの）


def active_purposes(cfg, dests, vehicles) -> list:
    """選んだ目的地・車種で使える用途。両端が選ばれ、使える車種が1つ以上あるもの。"""
    out = []
    for p in cfg["opt_time"]["purposes"]:
        ends = [x for x in (p["from"], p["to"]) if x != "house"]
        al = [TYPES.index(v) for v in p["vehicles"] if v in vehicles]
        if all(e in dests for e in ends) and al:
            out.append({**p, "allowed": tuple(al)})
    return out


def _dest_nodes(net, cfg, dests):
    """目的地ポートの徒歩ノード（node, snap_m）と、ポート→入口の徒歩（m）。"""
    pts = []
    for d in dests:
        p = cfg["destinations"][d].get("port") or cfg["opt_time"]["dest_ports"][d]
        pts.append({"name": d, "lat": p["lat"], "lon": p["lon"]})
    sp = snap_ports(net, pd.DataFrame(pts))
    egress = []
    for d, r in zip(dests, sp.itertuples()):
        if d in net.dest_keys:
            egress.append(float(net.E[net.dest_keys.index(d)]))
        else:
            egress.append(float(net.DnW[int(r.node), net.walk_dest_keys.index(d)] + r.snap_m))
    return sp, np.array(egress)


def _house_speed(cfg, d, allowed, leg=0) -> float:
    """住宅⇔目的地 d の実効走行速度（km/h）：使える車種に均等、経路別の実測（無ければ車種の一律速度）。"""
    if d in DESTS:
        rk = route_kmh(cfg)[DESTS.index(d)][leg]
        vs = [rk[t] for t in allowed]
    else:
        vs = [type_speed(cfg, TYPES[t]) for t in allowed]
    return 1.0 / np.mean([1 / v for v in vs])


def time_tensors(net, cfg, dests, purposes, proc_min):
    """(徒歩だけ (h, D), モビリティ (h, 候補, D), 候補ノード, 目的地ノード)。分。住宅の用途がある目的地だけ。"""
    vw = kmh_to_m_per_min(cfg["walk_speed_kmh"])
    cands = snap_ports(net, net.candidates[["lat", "lon"]].assign(name=""))
    dsp, egress = _dest_nodes(net, cfg, dests)
    hd = [p["to"] for p in purposes if p["from"] == "house"]
    wo = np.column_stack([walk_to(net, cfg, d) for d in hd]) / vw if hd else np.zeros((len(net.houses), 0))
    mob = np.zeros((len(net.houses), len(cands), len(hd)))
    cn = cands["node"].to_numpy()
    for i, d in enumerate(hd):
        p = next(q for q in purposes if q["from"] == "house" and q["to"] == d)
        j = dests.index(d)
        ride = (net.ride_nodes(cn, np.full(len(cn), int(dsp["node"].iloc[j]))) + cands["snap_m"].to_numpy()
                + float(dsp["snap_m"].iloc[j]))
        vr = kmh_to_m_per_min(_house_speed(cfg, d, p["allowed"]))
        mob[:, :, i] = (net.W / vw) + proc_min + (ride / vr)[None, :] + egress[j] / vw
    return wo, mob, cands, dsp, egress, hd


def _uniform(n, hours):
    v = np.zeros(24)
    if n > 0:
        h0, h1 = int(hours[0]), int(hours[1])
        v[h0:h1] = n / (h1 - h0)
    return v


def compute(net, cfg, survey, *, dests, vehicles, target, k_max, A, X, seed=0) -> dict:
    """①の計算一式。戻り値は画面・シートで使う値の辞書。"""
    cfg = copy.deepcopy(cfg)
    oc = cfg["opt_time"]
    cfg["procedure_sec"] = oc["procedure_sec"]
    for v in cfg["vehicles"].values():                 # ポートの上限は「台数」で数える（4輪も1台）
        v["slots"] = 1
    proc = oc["procedure_sec"] / 60
    purposes = active_purposes(cfg, dests, vehicles)
    if not purposes:
        return {"error": "選んだ目的地・車両で使える利用用途がありません。"}
    on = [t in vehicles for t in TYPES]
    mix = tuple((1.0 / sum(on) if o else 0.0) for o in on)
    wo, mob, cands, dsp, egress, hd = time_tensors(net, cfg, dests, purposes, proc)
    vw = kmh_to_m_per_min(cfg["walk_speed_kmh"])
    acc = ts.access_time(net, cfg)
    q = np.zeros((len(net.houses), 0), bool)
    S, assign_hd = [], np.full((len(net.houses), len(hd)), -1)
    if hd:
        mobl = ts.limit_access(mob, acc, A)
        share = np.ones(len(hd))
        S = ts.solve_max_cover(wo, mobl, k_max, X, share)
        q, saving = ts.coverage(wo, mobl, S, X, share)
        used = sorted({int(np.argmin(mobl[h, S, i])) for h, i in zip(*np.nonzero(q))})
        S = [S[u] for u in used]                       # 対象の住宅が使うポートだけ残す
        if S:
            q, saving = ts.coverage(wo, mobl, S, X, share)
            sub = mobl[:, S, :]
            assign_hd = np.where(q, sub.argmin(1), -1)
    k = len(S)
    if k == 0 and any(p["from"] == "house" for p in purposes) and not any(p["from"] != "house" for p in purposes):
        return {"error": f"ポートまで{A:g}分以内で、{X:g}分以上短くなる住宅がありません。条件をゆるめてください。",
                "n_cover": 0}
    k = max(k, 1)                                      # 住宅の用途が無くても配置の形のため1か所
    if not S:
        S = [int(np.argmin(acc.mean(0)))]
    cnode = cands["node"].to_numpy()[S]
    csnap = cands["snap_m"].to_numpy()[S]
    W = net.W[:, S]
    # 住宅ごとの代表ポート（色分け用）：対象の目的地のうち最初のもの
    any_q = (assign_hd >= 0).any(1) if hd else np.zeros(len(net.houses), bool)
    house_port = np.where(any_q, np.array([row[row >= 0][0] if (row >= 0).any() else -1 for row in assign_hd]), -1)
    w_all = np.bincount(house_port[house_port >= 0], minlength=k).astype(float)
    share_k = w_all / w_all.sum() if w_all.sum() > 0 else np.full(k, 1 / k)
    walk_k = np.array([W[house_port == j, j].mean() if (house_port == j).any() else W[:, j].mean() for j in range(k)])
    # ポート：住宅側 0..k-1、駅 k、イオン k+1、公園・エニタイム k+2..（DESTS は常に置く）
    dport = {"station": k, "aeon": k + 1}
    extra = [d for d in dests if d not in dport]
    for i, d in enumerate(extra):
        dport[d] = k + 2 + i
    cap = float(oc["port_cap"])
    caps = np.array([cap] * k + [np.inf, cap] + [cap] * len(extra))
    dnode = {d: (int(dsp["node"].iloc[dests.index(d)]), float(dsp["snap_m"].iloc[dests.index(d)])) for d in dests}
    eg = {d: float(egress[dests.index(d)]) for d in dests}
    for d in ("station", "aeon"):                       # 目的地に選ばなくても、ポートの番号は確保
        if d not in dnode:
            sp1, eg1 = _dest_nodes(net, cfg, [d])
            dnode[d], eg[d] = (int(sp1["node"].iloc[0]), float(sp1["snap_m"].iloc[0])), float(eg1[0])
    alt_k = np.zeros((k, 2))
    ridek = np.zeros((k, 2))
    for i, d in enumerate(DESTS):
        nd, sn = dnode[d]
        ridek[:, i] = net.ride_nodes(cnode, np.full(k, nd)) + csnap + sn
        wd = walk_to(net, cfg, d)
        alt_k[:, i] = [wd[house_port == j].mean() if (house_port == j).any() else wd.mean() for j in range(k)]
    flows, cols = [], {"平日": [], "休日": []}
    one = np.ones(k)
    n_cover = {}
    for p in purposes:
        a, b = p["from"], p["to"]
        dw = float(p.get("dwell_hours", 0) or 0) * 60
        if a == "house":
            i = hd.index(b)
            cnt = np.bincount(assign_hd[:, i][assign_hd[:, i] >= 0], minlength=k).astype(float)
            n = float(cnt.sum()) * oc["trips"]
            n_cover[p["label"]] = int(cnt.sum())
            if n <= 0:
                continue
            wts = cnt / cnt.sum()
            m = assign_hd[:, i]
            walk_j = np.array([W[m == j, j].mean() if (m == j).any() else walk_k[j] for j in range(k)])
            wd = walk_to(net, cfg, b)
            alt_j = np.array([wd[m == j].mean() if (m == j).any() else wd.mean() for j in range(k)])
            nd, sn = dnode[b]
            ride = net.ride_nodes(cnode, np.full(k, nd)) + csnap + sn
            for day in ("平日", "休日"):
                if p["days"] == "休日" and day != "休日":
                    shape_o = shape_r = np.zeros(24)
                elif b in DESTS:
                    o_, r_ = ts.demand_for(survey, cfg, {b: n}, day)
                    shape_o, shape_r = o_[:, DESTS.index(b)], r_[:, DESTS.index(b)]
                else:
                    to_, _r = table_hourly(cfg, b)           # 需要表の時間帯の形（無ければ設定の時間帯に一様）
                    shape_o = to_ / to_.sum() * n if to_.sum() > 0 else _uniform(n, p["hours"])
                    shape_r = np.zeros(24)
                if b == "station":                      # 駅は片道：朝の往路と夕方の復路を別の利用者として
                    cols[day] += [shape_o, shape_r]
                else:
                    cols[day] += [shape_o]
            base = dict(ride_m=ride, alt_m=alt_j, weights=wts, allowed=p["allowed"])
            if b == "station":
                flows.append(Flow(label=p["label"], origin=-1, dest=dport[b], round_trip=False, dwell_min=0,
                                  walk_o_m=walk_j, walk_d_m=one * eg[b], **base))
                flows.append(Flow(label=p["label"], origin=dport[b], dest=-1, round_trip=False, dwell_min=0,
                                  walk_o_m=one * eg[b], walk_d_m=walk_j, **base))
            else:
                if b == "aeon" and not dw:
                    dw = 60.0
                flows.append(Flow(label=p["label"], origin=-1, dest=dport[b], round_trip=True, dwell_min=dw,
                                  walk_o_m=walk_j, walk_d_m=one * eg[b], **base))
        else:                                           # 駅発（住宅が関わらない）：回数固定
            na, sa = dnode[a]
            nb_, sb = dnode[b]
            ride = float(net.ride_nodes([na], [nb_])[0]) + sa + sb
            alt = float(net.DW[net.all_dest_keys.index(a), net.all_dest_keys.index(b)])
            w0 = np.zeros(k)
            w0[0] = 1.0
            legs = [(a, b), (b, a)] if p.get("one_way") else [(a, b)]
            for (x, y) in legs:
                flows.append(Flow(label=p["label"], origin=dport[x], dest=dport[y], round_trip=not p.get("one_way"),
                                  dwell_min=dw, ride_m=one * ride, walk_o_m=one * eg[x], walk_d_m=one * eg[y],
                                  alt_m=one * alt, weights=w0, allowed=p["allowed"]))
                for day in ("平日", "休日"):
                    on_day = not (p["days"] == "休日" and day != "休日")
                    cols[day].append(_uniform(float(p["per_day"]) if on_day else 0.0, p["hours"]))
    if not flows:
        return {"error": "条件を満たす利用がありません。条件をゆるめてください。", "n_cover": 0}
    L = Layout(k=k, share=share_k, walk_m=walk_k, ride_m=ridek, egress_m=np.array([eg["station"], eg["aeon"]]),
               alt_m=alt_k, caps=caps, labels=[f"P{j + 1}" for j in range(k)], flows=flows)
    dem = {}
    for day in ("平日", "休日"):
        fc = np.column_stack(cols[day])
        dem[day] = (np.hstack([np.zeros((24, 2)), fc]), np.zeros((24, 2 + fc.shape[1])))
    kw = dict(pattern=oc["pattern"], substitution=oc["substitution"], ebike_share=mix, procedure_sec=oc["procedure_sec"],
              route_speeds=oc["route_speeds"])
    year = int(cfg["month"]["year"])
    cal, specs = [], []
    src = set()
    for mth in oc["months"]:
        c_, s_, so = month_specs(cfg, year, int(mth), dem, kw, seed)
        cal.append(c_)
        specs += s_
        src.add(so)
    cal = pd.concat(cal, ignore_index=True)
    ops = make_ops(cfg, weather="晴れ", **kw)
    t = round(1 - target, 4)
    # 上限の確認：すべてのポートを上限まで埋め、駅にも十分置いたときの充足率（これが目標に届かなければ達成不可）
    peak = peak_outflow(*dem["平日"])
    n_big = int(np.nansum(np.where(np.isfinite(caps), caps, 0)) + np.ceil(peak * 0.3) + 4)   # 駅には余裕分だけ
    split = [n_big // sum(on) if o else 0 for o in on]
    split[[i for i, o in enumerate(on) if o][0]] += n_big - sum(split)
    F_big = allocate_fleet(L, ops, split[0], split[1], *dem["平日"], n_tr=split[2], seed=seed)
    f_, q_ = 0, 0
    for i, spec in enumerate(specs):
        d_, _ = simulate_day(L, F_big, *spec, np.random.default_rng([seed, i]))
        f_, q_ = f_ + d_.fail_out + d_.fail_ret, q_ + d_.req_out + d_.req_ret
    best_possible = 1 - f_ / q_ if q_ else 1.0
    if best_possible < target:
        r, F = None, F_big
    else:
        res = min_fleet(L, ops, *dem["平日"], targets=(t,), n_rep=len(specs), seed=seed, day_specs=specs, types_on=on)
        r = res[t]
        F = F_big if r is None else None
    out = dict(best_possible=best_possible, L=L, S=S, cands=cands, assign=house_port, assign_hd=assign_hd, hd=hd, q=q, purposes=purposes,
               dests=dests, vehicles=vehicles, dport=dport, dnode=dnode, n_cover=n_cover, cal=cal,
               source="・".join(sorted(src)), target=target, A=A, X=X, k_max=k_max, sel=r,
               n_houses=len(net.houses), n_hh=int(any_q.sum()), cfg=cfg)
    if r is not None:
        F = fleet_for(L, r["自転車"], r["キックボード"], *dem["平日"], ops, n_tr=r.get("4輪", 0))
    else:                                       # 達成不可：すべてのポートを上限まで埋めた台数での結果（参考）
        out["ref"] = {"自転車": int(F.init[:, 0].sum()), "キックボード": int(F.init[:, 1].sum()),
                      "4輪": int(F.init[:, 2].sum()), "合計": int(F.init.sum())}
    ms, daily = month_eval(L, F, cfg, dem, seed=seed + 1, cal=cal, **kw)
    peaks = ms["ポート別最大駐車(95%点)"]
    out.update(init=F.init, ms=ms, daily=daily, station_need=int(peaks[k]),
               cost=annual_cost(cfg, F, k, daily, int(peaks[k])),
               saving=ms["全員徒歩の平均(分/片道)"] - ms["平均移動時間(分/片道)"])
    return out


__all__ = ["DEST_ALL", "DEST_LABEL", "TYPE_LABEL", "active_purposes", "compute", "month_calendar"]
