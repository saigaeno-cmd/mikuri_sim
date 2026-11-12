"""条件ごとの結果シート（1条件＝1ページ、16:9）。②ポート配置評価と同じ並び：
最適ポート配置（ポート数・設置台数・充足率・合計移動時間）→ 左：ポート場所の地図／右：時間帯別の利用希望と充足・ポートごとの表。

compute_time_opt：①ポート最適化（時間短縮）を画面の既定値で計算する（画面と同じ手順）。
"""
from __future__ import annotations

import copy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from . import present as _pr  # noqa: E402,F401（日本語フォントの設定）
from . import timesave as ts  # noqa: E402
from .demand import DESTS  # noqa: E402
from .engine import fleet_label  # noqa: E402
from .plan import default_mix, demand_hourly, fleet_for, make_ops  # noqa: E402
from .present import GRAY, INK, LINE, MUTED, NAVY, ORANGE, PORT_COLORS, TEAL, to_png, to_pptx  # noqa: E402,F401
from .study import annual_cost, min_fleet, month_eval  # noqa: E402
from .ui import port_table  # noqa: E402
from .weather_data import available_months, month_specs  # noqa: E402


# ---------------------------------------------------------------- 計算
def defaults(cfg) -> dict:
    """画面（①）の既定値。"""
    ms = available_months(cfg)
    year, month = ms[0] if ms else (int(cfg["month"]["year"]), int(cfg["month"]["month"]))
    return dict(trips=1.0, mix=default_mix(cfg), substitution=cfg["fleet_mix"]["substitution"], pattern="a",
                station_cap=int(cfg["station_port"]["max_slots"]), aeon_cap=int(cfg["service"].get("aeon_max_slots") or 10),
                procedure_sec=float(cfg["procedure_sec"]), year=year, month=month, passes=2, route_speeds=True,
                target=float(cfg["service"]["target_fulfillment"]), station_share=None)


def compute_time_opt(net, cfg, survey, *, A: float, X: float, k: int, **kw) -> dict:
    """①ポート最適化（時間短縮）。A：住宅→ポートの徒歩（分以内）、X：短くなる時間（分以上）、k：住宅側ポート数。
    kw は defaults() の項目を上書き。"""
    cfg = copy.deepcopy(cfg)
    p = {**defaults(cfg), **kw}
    cfg["service"]["aeon_max_slots"] = int(p["aeon_cap"])
    P = cfg["survey_xlsx"]["default_modes"]
    _, _, sh = demand_hourly(survey, cfg, 1.0, P, "平日")
    st_share = int(round(sh[0] * 100)) / 100 if p["station_share"] is None else p["station_share"]
    share = np.array([st_share, 1 - st_share])
    wo, mob = ts.time_tensors(net, cfg, p["mix"], procedure_sec=p["procedure_sec"], route_speeds=p["route_speeds"])
    acc = ts.access_time(net, cfg)
    mob = ts.limit_access(mob, acc, A)
    S = ts.solve_max_cover(wo, mob, k, X, share)
    q, saving = ts.coverage(wo, mob, S, X, share)
    L, assign, _ = ts.layout_for(net, cfg, S, q, share, wo, mob, station_cap=p["station_cap"])
    n_trip = {key: float((q[:, d] * share[d]).sum() * p["trips"]) for d, key in enumerate(DESTS)}
    out = dict(params=p, A=A, X=X, k=k, share=share, S=S, q=q, saving=saving, assign=assign, L=L, n_trip=n_trip,
               n_hh=int(q.any(1).sum()), n_houses=len(net.houses), houses=net.houses, candidates=net.candidates,
               cfg=cfg, sel=None)
    if sum(n_trip.values()) <= 0:
        return out
    dem = {dt: ts.demand_for(survey, cfg, n_trip, dt) for dt in ("平日", "休日")}
    kw_ops = dict(pattern=p["pattern"], substitution=p["substitution"], ebike_share=p["mix"],
                  procedure_sec=p["procedure_sec"], route_speeds=p["route_speeds"])
    cal, specs, source = month_specs(cfg, p["year"], p["month"], dem, kw_ops)
    ops = make_ops(cfg, weather="晴れ", **kw_ops)
    t = round(1 - p["target"], 4)
    res = min_fleet(L, ops, *dem["平日"], targets=(t,), n_rep=len(specs) * p["passes"], seed=0, day_specs=specs)
    r = res[t]
    out.update(cal=cal, source=source, sel=r)
    if r is None:
        return out
    F = fleet_for(L, r["自転車"], r["キックボード"], *dem["平日"], ops, n_tr=r.get("4輪", 0))
    ms, daily = month_eval(L, F, cfg, dem, month=p["month"], seed=1, cal=cal, **kw_ops)
    out.update(init=F.init, ms=ms, cost=annual_cost(cfg, F, L.k, daily, p["station_cap"]))
    return out


# ---------------------------------------------------------------- 描画
def _roads(net_dir):
    """徒歩道路網の線（地図の背景）。"""
    import networkx as nx
    g = nx.read_graphml(net_dir / "walk.graphml")
    xs, ys = [], []
    for u, v in g.edges():
        a, b = g.nodes[u], g.nodes[v]
        xs += [float(a["x"]), float(b["x"]), np.nan]
        ys += [float(a["y"]), float(b["y"]), np.nan]
    return np.array(xs), np.array(ys)


def _box(fig, rect, label, value, sub=""):
    ax = fig.add_axes(rect)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.04), 0.98, 0.92, boxstyle="round,pad=0.0,rounding_size=0.12",
                                fc="white", ec=INK, lw=1.6, transform=ax.transAxes))
    ax.text(0.5, 0.74, label, fontsize=17, color=INK, ha="center", va="center", transform=ax.transAxes)
    ax.text(0.5, 0.36, value, fontsize=27, weight="bold", color=NAVY, ha="center", va="center", transform=ax.transAxes)
    if sub:
        ax.text(0.5, 0.12, sub, fontsize=10.5, color=MUTED, ha="center", va="center", transform=ax.transAxes)


def _map(ax, c, roads):
    L, init, cfg = c["L"], c.get("init"), c["cfg"]
    h = c["houses"]
    if roads is not None:
        ax.plot(roads[0], roads[1], color="#D5DAE1", lw=0.8, zorder=1)
    a = np.asarray(c["assign"])
    cols = [PORT_COLORS[x % len(PORT_COLORS)] if x >= 0 else "#B9BFC9" for x in a]
    ax.scatter(h["lon"], h["lat"], s=14, c=cols, alpha=[0.9 if x >= 0 else 0.45 for x in a], lw=0, zorder=2)
    cand = c["candidates"].loc[c["S"]]
    for j, (_, p) in enumerate(cand.iterrows()):
        col = PORT_COLORS[j % len(PORT_COLORS)]
        ax.scatter(p["lon"], p["lat"], s=360, marker="s", c=col, edgecolors="white", lw=2, zorder=4)
        ax.annotate(f"P{j + 1}", (p["lon"], p["lat"]), color="white", ha="center", va="center", fontsize=10.5,
                    weight="bold", zorder=5)
        if init is not None:
            ax.annotate(fleet_label(init[j]), (p["lon"], p["lat"]), xytext=(12, -16), textcoords="offset points",
                        fontsize=10, color=INK, zorder=5, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=.9))
    pts = [h[["lon", "lat"]].to_numpy()]
    for d, row in zip(("station", "aeon"), (L.k, L.k + 1)):
        dd = cfg["destinations"][d]
        p = dd["port"]
        n = int(init[row].sum()) if init is not None else 0
        cap = c["params"]["station_cap"] if d == "station" else c["params"]["aeon_cap"]
        ax.scatter(p["lon"], p["lat"], s=260, marker="D", c=ORANGE if d == "aeon" else NAVY, edgecolors="white", lw=2,
                   zorder=4)
        ax.annotate(f"{dd['port_label']}（初期{n}台・収納{cap}）", (p["lon"], p["lat"]), xytext=(9, 7),
                    textcoords="offset points", fontsize=10.5, color=INK, weight="bold", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=.85))
        pts.append([[p["lon"], p["lat"]]])
    pts = np.vstack(pts)
    lat = pts[:, 1]
    ax.set_xlim(pts[:, 0].min() - 0.0007, pts[:, 0].max() + 0.0022)
    ax.set_ylim(lat.min() - 0.0005, lat.max() + 0.0006)
    ax.set_aspect(1 / np.cos(np.radians(lat.mean())))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(LINE)


def _hourly(ax, hr):
    hr = np.array(hr)
    hrs = np.arange(24)
    used, req = hr[:, 1], hr[:, 0]
    ax.bar(hrs, used, color=NAVY, width=0.75, label="利用できた")
    ax.bar(hrs, req - used, bottom=used, color=GRAY, width=0.75, label="利用できなかった")
    ax.set_xlim(4.5, 23.5)
    ax.set_xticks(range(5, 24))
    ax.tick_params(labelsize=9, colors=MUTED)
    ax.set_xlabel("時間帯（時）", fontsize=10, color=MUTED)
    ax.set_ylabel("利用希望（件/日）", fontsize=10, color=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#C9CED6")
    ax.grid(axis="y", color="#EEF0F3")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, frameon=False, loc="upper left", ncol=2)


def _table(ax, df):
    ax.axis("off")
    cols = list(df.columns)
    t = ax.table(cellText=df.astype(str).values, colLabels=cols, loc="upper center", cellLoc="center",
                 colLoc="center", bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False)
    t.set_fontsize(9.5)
    for (r, cidx), cell in t.get_celld().items():
        cell.set_edgecolor("#DDE1E7")
        if r == 0:
            cell.set_facecolor("#F3F5F8")
            cell.set_text_props(color=MUTED, weight="bold")
    return t


def sheet_time_opt(c, title: str, cond: str, roads=None):
    """①の結果シート（16:9の Figure）。"""
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    fig.text(0.03, 0.95, title, fontsize=24, weight="bold", color=INK, va="center")
    fig.text(0.03, 0.905, cond, fontsize=11, color=MUTED, va="center")
    fig.text(0.03, 0.855, "最適ポート配置", fontsize=18, weight="bold", color=INK, va="center")
    p, sel, ms = c["params"], c.get("sel"), c.get("ms")
    w, gap, x0 = 0.225, 0.0133, 0.03
    vals = [("ポート数", f"{c['L'].k} か所", f"対象の住宅 {c['n_hh']}/{c['n_houses']}戸"),
            ("設置台数", f"{sel['合計']} 台" if sel else "達成不可",
             (f"自転車{sel['自転車']}・キックボード{sel['キックボード']}・4輪{sel.get('4輪', 0)}" if sel else "")),
            ("充足率", f"{ms['充足率']:.1%}" if ms else "—", f"目標 {p['target']:.0%}以上（1か月）"),
            ("合計移動時間", f"{ms['合計移動時間(人・分/日)']:,.0f} 人・分/日" if ms else "—",
             (f"平均 {ms['平均移動時間(分/片道)']:.1f}分/片道（全員徒歩 {ms['全員徒歩の平均(分/片道)']:.1f}分）" if ms else ""))]
    for i, (lab, val, sub) in enumerate(vals):
        _box(fig, [x0 + i * (w + gap), 0.69, w, 0.14], lab, val, sub)
    fig.text(0.03, 0.655, "ポート場所", fontsize=18, weight="bold", color=INK, va="center")
    _map(fig.add_axes([0.03, 0.035, 0.55, 0.6]), c, roads)
    fig.text(0.03, 0.018, "色つきの点＝対象の住宅（色は使うポート）、灰色＝ポートが遠い・短縮が足りず利用しない住宅。"
             "ラベル：自＝電動アシスト自転車、キ＝電動キックボード、4輪＝4輪小型モビリティの初期台数",
             fontsize=8.5, color=MUTED)
    fig.text(0.61, 0.655, "時間帯別の利用希望と充足", fontsize=13, weight="bold", color=INK, va="center")
    if ms:
        _hourly(fig.add_axes([0.645, 0.4, 0.33, 0.235]), ms["_hourly"])
        fig.text(0.61, 0.335, "ポートごと", fontsize=13, weight="bold", color=INK, va="center")
        L = c["L"]
        bp = port_table([f"P{j + 1}" for j in range(L.k)], [None] * L.k, c["init"], L.caps, L.k,
                        ms["ポート別最大駐車(95%点)"], c["assign"], p["station_cap"])
        bp["最大駐車(95%点)"] = [int(x) for x in bp["最大駐車(95%点)"]]
        bp = bp.rename(columns={"最大駐車(95%点)": "最大駐車(95%)", "キックボード": "キック"})
        _table(fig.add_axes([0.61, 0.035, 0.365, 0.28]), bp)
    else:
        fig.text(0.61, 0.5, "目標を満たす台数が見つかりませんでした（条件を満たす住宅が無い、または探索範囲を超える）。",
                 fontsize=12, color=INK, wrap=True)
    return fig
