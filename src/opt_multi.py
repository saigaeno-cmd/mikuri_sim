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


def _snap_cands(net, cands=None):
    """ポート候補（無ければ事前計算の候補地点）を徒歩ノードに合わせ、(候補表, 住宅→候補の徒歩 (h, n))。"""
    if cands is None:
        c = snap_ports(net, net.candidates[["lat", "lon"]].assign(name=""))
        return c, net.W
    c = snap_ports(net, cands)
    return c, net.Wn[:, c["node"].to_numpy()] + c["snap_m"].to_numpy()[None, :]


def time_tensors(net, cfg, dests, purposes, proc_min, cands=None):
    """(徒歩だけ (h, D), モビリティ (h, 候補, D), 候補表, 目的地ノード, 施設側徒歩, 住宅の用途の目的地, 住宅→候補の徒歩)。
    時間は分。D は住宅の用途がある目的地。"""
    vw = kmh_to_m_per_min(cfg["walk_speed_kmh"])
    cands, Wc = _snap_cands(net, cands)
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
        mob[:, :, i] = (Wc / vw) + proc_min + (ride / vr)[None, :] + egress[j] / vw
    return wo, mob, cands, dsp, egress, hd, Wc


def _uniform(n, hours):
    v = np.zeros(24)
    if n > 0:
        h0, h1 = int(hours[0]), int(hours[1])
        v[h0:h1] = n / (h1 - h0)
    return v


def prepare(cfg, dests, vehicles):
    """固定条件を入れた設定・有効な用途・車種の比率・運用条件。"""
    cfg = copy.deepcopy(cfg)
    oc = cfg["opt_time"]
    cfg["procedure_sec"] = oc["procedure_sec"]
    for v in cfg["vehicles"].values():                 # ポートの上限は「台数」で数える（4輪も1台）
        v["slots"] = 1
    purposes = active_purposes(cfg, dests, vehicles)
    on = [t in vehicles for t in TYPES]
    mix = tuple((1.0 / sum(on) if o else 0.0) for o in on) if any(on) else (1.0, 0.0, 0.0)
    kw = dict(pattern=oc["pattern"], substitution=oc["substitution"], ebike_share=mix, procedure_sec=oc["procedure_sec"],
              route_speeds=oc["route_speeds"])
    return cfg, oc, purposes, on, kw


def build(net, cfg, survey, *, dests, purposes, hd, cnode, csnap, W, assign_hd, weight_hd, house_caps, dest_caps):
    """選んだ住宅側ポート（k か所）での Layout と、曜日区分ごとの需要 {曜日: (out_h, ret_h)}。
    assign_hd：(住宅, 住宅の用途の目的地) 使うポート（-1＝使わない）。weight_hd：その組の1日の往復数。
    dest_caps：{目的地: 収納（np.inf＝上限なし）}。戻り値 (L, dem, 住宅の代表ポート, 用途ごとの住宅数)。"""
    k = len(cnode)
    any_q = (assign_hd >= 0).any(1) if hd else np.zeros(len(net.houses), bool)
    house_port = np.where(any_q, np.array([row[row >= 0][0] if (row >= 0).any() else -1 for row in assign_hd]), -1)
    w_all = np.bincount(house_port[house_port >= 0], minlength=k).astype(float)
    share_k = w_all / w_all.sum() if w_all.sum() > 0 else np.full(k, 1 / k)
    walk_k = np.array([W[house_port == j, j].mean() if (house_port == j).any() else W[:, j].mean() for j in range(k)])
    dport = {"station": k, "aeon": k + 1}
    extra = [d for d in dests if d not in dport]
    for i, d in enumerate(extra):
        dport[d] = k + 2 + i
    caps = np.array(list(house_caps) + [dest_caps.get("station", np.inf), dest_caps.get("aeon", np.inf)]
                    + [dest_caps.get(d, np.inf) for d in extra], float)
    dsp, egress = _dest_nodes(net, cfg, dests)
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
            m = assign_hd[:, i]
            cnt = np.bincount(m[m >= 0], weights=weight_hd[m >= 0, i], minlength=k).astype(float)
            n = float(cnt.sum())
            n_cover[p["label"]] = int((m >= 0).sum())
            if n <= 0:
                continue
            wts = cnt / n
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
                cols[day] += [shape_o, shape_r] if b == "station" else [shape_o]   # 駅は片道：朝と夕方を別の利用者
            base = dict(ride_m=ride, alt_m=alt_j, weights=wts, allowed=p["allowed"])
            if b == "station":
                flows.append(Flow(label=p["label"], origin=-1, dest=dport[b], round_trip=False, dwell_min=0,
                                  walk_o_m=walk_j, walk_d_m=one * eg[b], **base))
                flows.append(Flow(label=p["label"], origin=dport[b], dest=-1, round_trip=False, dwell_min=0,
                                  walk_o_m=one * eg[b], walk_d_m=walk_j, **base))
            else:
                flows.append(Flow(label=p["label"], origin=-1, dest=dport[b], round_trip=True,
                                  dwell_min=dw or 60.0, walk_o_m=walk_j, walk_d_m=one * eg[b], **base))
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
        return None, None, house_port, n_cover
    L = Layout(k=k, share=share_k, walk_m=walk_k, ride_m=ridek, egress_m=np.array([eg["station"], eg["aeon"]]),
               alt_m=alt_k, caps=caps, labels=[f"P{j + 1}" for j in range(k)], flows=flows)
    dem = {}
    for day in ("平日", "休日"):
        fc = np.column_stack(cols[day])
        dem[day] = (np.hstack([np.zeros((24, 2)), fc]), np.zeros((24, 2 + fc.shape[1])))
    return L, dem, house_port, n_cover


def period(cfg, oc, dem, kw, seed=0):
    """検証期間（opt_time.months の3か月）の暦と、日ごとの（運用条件, 需要）。"""
    year = int(cfg["month"]["year"])
    cal, specs, src = [], [], set()
    for mth in oc["months"]:
        c_, s_, so = month_specs(cfg, year, int(mth), dem, kw, seed)
        cal.append(c_)
        specs += s_
        src.add(so)
    return pd.concat(cal, ignore_index=True), specs, "・".join(sorted(src))


def bound(L, ops, dem, specs, on, seed=0):
    """すべてのポートを上限まで埋め、駅にも余裕分を置いたときの充足率（上限の目安）と、その配置。"""
    peak = peak_outflow(*dem["平日"])
    n_big = int(np.nansum(np.where(np.isfinite(L.caps), L.caps, 0)) + np.ceil(peak * 0.3) + 4)
    split = [n_big // sum(on) if o else 0 for o in on]
    split[[i for i, o in enumerate(on) if o][0]] += n_big - sum(split)
    F = allocate_fleet(L, ops, split[0], split[1], *dem["平日"], n_tr=split[2], seed=seed)
    f_, q_, t_, w_, n_ = 0, 0, 0.0, 0.0, 0
    for i, spec in enumerate(specs):
        d_, _ = simulate_day(L, F, *spec, np.random.default_rng([seed, i]))
        f_, q_ = f_ + d_.fail_out + d_.fail_ret, q_ + d_.req_out + d_.req_ret
        t_, w_, n_ = t_ + d_.time_min, w_ + d_.walk_only_min, n_ + d_.legs
    return (1 - f_ / q_ if q_ else 1.0), F, (t_ / n_ if n_ else 0.0)


def _split(n, weights, on):
    """合計 n 台を車種の重み（使わない車種は0）で整数に分ける。"""
    w = np.array([max(float(x), 0.0) if o else 0.0 for x, o in zip(weights, on)])
    if w.sum() <= 0:
        w = np.array([1.0 if o else 0.0 for o in on])
    raw = n * w / w.sum()
    out = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - out))[: n - out.sum()]:
        out[i] += 1
    return [int(x) for x in out]


def finish(net, cfg, oc, L, dem, cal, specs, kw, on, target, seed=0, search=True, max_by_type=None) -> dict:
    """目標充足率を満たす最小台数（上限まで埋めても届かなければ達成不可と参考の結果）と、その3か月の評価。
    max_by_type：車種ごとの上限台数 (自転車, キックボード, 4輪)（None＝上限なし）。上限の範囲で目標を満たせないときは、
    上限の台数（上限の無い車種は、ポートを上限まで埋めたときの台数）での結果を参考に出す。"""
    caps_t = list(max_by_type or [None, None, None])
    has_cap = any(c is not None for c, o in zip(caps_t, on) if o)
    ops = make_ops(cfg, weather="晴れ", **kw)
    t = round(1 - target, 4)
    best_possible, F_big, _ = bound(L, ops, dem, specs, on, seed)
    r, over = None, None
    if best_possible >= target and search:
        r = min_fleet(L, ops, *dem["平日"], targets=(t,), n_rep=len(specs), seed=seed, day_specs=specs, types_on=on,
                      max_by_type=caps_t if has_cap else None)[t]
    out = dict(best_possible=best_possible, sel=r, over=over, max_by_type=caps_t if has_cap else None)
    if r is not None:
        F = fleet_for(L, r["自転車"], r["キックボード"], *dem["平日"], ops, n_tr=r.get("4輪", 0))
    elif has_cap:
        # 車種ごとの上限の台数で配置（上限の無い車種は、ポートを上限まで埋めたときの台数）
        big = [int(F_big.init[:, i].sum()) for i in range(3)]
        n3 = [(0 if not o else big[i] if c is None else int(c)) for i, (c, o) in enumerate(zip(caps_t, on))]
        F = fleet_for(L, n3[0], n3[1], *dem["平日"], ops, n_tr=n3[2])
        out["ref"] = {"自転車": int(F.init[:, 0].sum()), "キックボード": int(F.init[:, 1].sum()),
                      "4輪": int(F.init[:, 2].sum()), "合計": int(F.init.sum()), "上限": True}
    else:                                       # 達成不可：すべてのポートを上限まで埋めた台数での結果（参考）
        F = F_big
        out["ref"] = {"自転車": int(F.init[:, 0].sum()), "キックボード": int(F.init[:, 1].sum()),
                      "4輪": int(F.init[:, 2].sum()), "合計": int(F.init.sum())}
    ms, daily = month_eval(L, F, cfg, dem, seed=seed + 1, cal=cal, **kw)
    k = L.k
    if np.isfinite(L.caps[k]):                  # 駅ポートに上限があるとき：上限なしで回した最大駐車を必要容量に
        caps = L.caps.copy()
        caps[k] = np.inf
        import dataclasses
        ms_inf, _ = month_eval(dataclasses.replace(L, caps=caps), F, cfg, dem, seed=seed + 1, cal=cal, **kw)
        need = int(ms_inf["ポート別最大駐車(95%点)"][k])
    else:
        need = int(ms["ポート別最大駐車(95%点)"][k])
    out.update(init=F.init, ms=ms, daily=daily, station_need=need,
               cost=annual_cost(cfg, F, k, daily, need if not np.isfinite(L.caps[k]) else int(L.caps[k])),
               saving=ms["全員徒歩の平均(分/片道)"] - ms["平均移動時間(分/片道)"])
    return out


def compute(net, cfg, survey, *, dests, vehicles, target, k_max, A, X, max_by_type=None, seed=0) -> dict:
    """①の計算一式。戻り値は画面・シートで使う値の辞書。"""
    cfg, oc, purposes, on, kw = prepare(cfg, dests, vehicles)
    if not purposes:
        return {"error": "選んだ目的地・車両で使える利用用途がありません。"}
    proc = oc["procedure_sec"] / 60
    wo, mob, cands, dsp, egress, hd, Wc = time_tensors(net, cfg, dests, purposes, proc)
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
            assign_hd = np.where(q, mobl[:, S, :].argmin(1), -1)
    if not S and not any(p["from"] != "house" for p in purposes):
        return {"error": f"ポートまで{A:g}分以内で、{X:g}分以上短くなる住宅がありません。条件をゆるめてください。",
                "n_cover": 0}
    if not S:                                          # 住宅の用途が無くても配置の形のため1か所
        S = [int(np.argmin(acc.mean(0)))]
    cap = float(oc["port_cap"])
    L, dem, house_port, n_cover = build(
        net, cfg, survey, dests=dests, purposes=purposes, hd=hd, cnode=cands["node"].to_numpy()[S],
        csnap=cands["snap_m"].to_numpy()[S], W=Wc[:, S], assign_hd=assign_hd,
        weight_hd=np.full(assign_hd.shape, float(oc["trips"])), house_caps=[cap] * len(S),
        dest_caps={"station": np.inf, "aeon": cap, "park": cap, "anytime": cap})
    if L is None:
        return {"error": "条件を満たす利用がありません。条件をゆるめてください。", "n_cover": 0}
    cal, specs, src = period(cfg, oc, dem, kw, seed)
    out = finish(net, cfg, oc, L, dem, cal, specs, kw, on, target, seed, max_by_type=max_by_type)
    out.update(L=L, S=S, cands=cands, assign=house_port, assign_hd=assign_hd, hd=hd, q=q, purposes=purposes,
               dests=dests, vehicles=vehicles, n_cover=n_cover, cal=cal, source=src, target=target, A=A, X=X,
               k_max=k_max, n_houses=len(net.houses), n_hh=int((house_port >= 0).sum()), cfg=cfg)
    return out


def evaluate_candidates(net, cfg, survey, cands: pd.DataFrame, *, dests, vehicles, target, k_max, conversion,
                        dest_caps: dict, max_combos: int = 300, seed=0, progress=None) -> dict:
    """②：候補地（name, lat, lon, cap）から上限 k_max か所以内の組み合わせを選び、最小台数を求める。
    需要：住宅すべてが目的地ごとに1日1往復 × 転換率（期待値）。住宅は所要時間が最短のポートを使う。
    組み合わせは「すべてのポートを容量まで埋めたときの充足率」が最大のもの（同じならポートが少なく、移動時間が短い方）。"""
    import itertools
    cfg, oc, purposes, on, kw = prepare(cfg, dests, vehicles)
    if not purposes:
        return {"error": "選んだ目的地・車両で使える利用用途がありません。"}
    if not len(cands):
        return {"error": "ポート候補地がありません。地図をダブルクリックして置いてください。"}
    proc = oc["procedure_sec"] / 60
    wo, mob, cs, dsp, egress, hd, Wc = time_tensors(net, cfg, dests, purposes, proc, cands=cands)
    caps_all = cands["cap"].to_numpy(float)
    trips = float(oc["trips"]) * conversion
    n = len(cands)
    m = min(int(k_max), n)
    combos = [c for r in range(1, m + 1) for c in itertools.combinations(range(n), r)]
    method = f"{len(combos)}通りを比較"
    if len(combos) > max_combos:                     # 多いときは1か所ずつ最良を足す
        combos, method = None, f"組み合わせが多いため、1か所ずつ最良を足す方法で探索"
    ops = make_ops(cfg, weather="晴れ", **kw)
    cache = {}

    def score(idx):
        idx = tuple(sorted(idx))
        if idx in cache:
            return cache[idx]
        S = list(idx)
        ahd = mob[:, S, :].argmin(1) if hd else np.zeros((len(net.houses), 0), int)
        L, dem, hp, nc = build(net, cfg, survey, dests=dests, purposes=purposes, hd=hd,
                               cnode=cs["node"].to_numpy()[S], csnap=cs["snap_m"].to_numpy()[S], W=Wc[:, S],
                               assign_hd=ahd, weight_hd=np.full(ahd.shape, trips), house_caps=caps_all[S],
                               dest_caps=dest_caps)
        if L is None:
            cache[idx] = (-1.0, 0.0, None)
            return cache[idx]
        cal, specs, src = period(cfg, oc, dem, kw, seed)
        bp, _, tt = bound(L, ops, dem, specs, on, seed)
        cache[idx] = (bp, tt, (L, dem, hp, nc, cal, specs, src, S, ahd))
        return cache[idx]

    rows = []
    if combos is not None:
        for i, c in enumerate(combos):
            bp, tt, _ = score(c)
            rows.append((c, bp, tt))
            if progress:
                progress((i + 1) / len(combos))
    else:
        chosen = []
        for step in range(m):
            cand_rows = [(tuple(chosen + [j]),) + score(chosen + [j])[:2] for j in range(n) if j not in chosen]
            best = max(cand_rows, key=lambda r: (r[1], -r[2]))
            if chosen and best[1] <= max(r[1] for r in rows) + 1e-9:
                break                                  # 足しても良くならない
            chosen = list(best[0])
            rows += cand_rows
            if progress:
                progress((step + 1) / m)
    best = max(rows, key=lambda r: (round(r[1], 3), -len(r[0]), -r[2]))
    L, dem, hp, nc, cal, specs, src, S, ahd = score(best[0])[2]
    out = finish(net, cfg, oc, L, dem, cal, specs, kw, on, target, seed)
    table = pd.DataFrame([{"使うポート": "・".join(str(cands["name"].iloc[j]) for j in r[0]), "ポート数": len(r[0]),
                           "充足率（容量まで埋めたとき）": r[1], "平均移動時間(分/片道)": r[2]} for r in rows])
    table = table.sort_values(["充足率（容量まで埋めたとき）", "ポート数", "平均移動時間(分/片道)"],
                              ascending=[False, True, True]).reset_index(drop=True)
    out.update(L=L, S=S, cands=cs, assign=hp, assign_hd=ahd, hd=hd, purposes=purposes, dests=dests, vehicles=vehicles,
               n_cover=nc, cal=cal, source=src, target=target, k_max=k_max, conversion=conversion,
               n_houses=len(net.houses), n_hh=int((hp >= 0).sum()), cfg=cfg, method=method, table=table,
               chosen=[str(cands["name"].iloc[j]) for j in S])
    return out


__all__ = ["DEST_ALL", "DEST_LABEL", "TYPE_LABEL", "active_purposes", "compute", "evaluate_candidates",
           "month_calendar"]
