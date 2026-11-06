"""発表用の6画面（16:9）。計算（compute）と描画（fig_*）を分け、画面とスクリプトの両方から使う。

1 対象地域　2 需要　3 最適配置　4 台数と充足率　5 実証（台数固定）　6 感度分析
③と④は同じ計算（p-median の配置＋1か月の暦で目標ごとに最小台数を探索）から作るので、数値がそろう。
"""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from .plan import demand_hourly, fleet_for, make_ops, optimal_layout  # noqa: E402
from .study import min_fleet, month_eval  # noqa: E402
from .weather_data import month_specs  # noqa: E402

NAVY, TEAL, PURPLE, GRAY, INK, MUTED, BG, ORANGE, LINE = (
    "#1A2F81", "#0F8A7E", "#6B5CA5", "#B8BEC9", "#111827", "#5B6475", "#F5F6F8", "#E07A1F", "#E2E5EA")
PORT_COLORS = [NAVY, TEAL, PURPLE, "#B5651D", "#2E7D32", "#8E24AA", "#00838F", "#6D4C41"]
TITLES = ["対象地域", "需要", "最適配置", "台数と充足率", "実証（台数固定）", "感度分析"]
CURVE_FAIL = (0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01)


def _font():
    """日本語フォント（Windows：游ゴシック・メイリオ、Mac：ヒラギノ、Linux：IPA・Noto）。"""
    names = {f.name for f in font_manager.fontManager.ttflist}
    for n in ("Yu Gothic", "Meiryo", "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Noto Sans CJK JP",
              "IPAexGothic", "IPAGothic", "MS Gothic"):
        if n in names:
            return n
    for f in font_manager.findSystemFonts():
        if "ipag" in f.lower() or "ipaexg" in f.lower():
            font_manager.fontManager.addfont(f)
            return font_manager.FontProperties(fname=f).get_name()
    return None


_F = _font()
if _F:
    plt.rcParams["font.family"] = _F


# ---------------------------------------------------------------- 計算
def compute(net, cfg, survey, *, conversion, year, month, station_cap, substitution, k=3, target=0.8,
            passes=2, seed=0, ebike_share=None) -> dict:
    """③④で使う計算。目標ごとの最小台数（曲線）と、目標 target での案。"""
    es = cfg["fleet_mix"]["ebike_share"] if ebike_share is None else ebike_share
    P = cfg["survey_xlsx"]["default_modes"]
    oh, rh, sh = demand_hourly(survey, cfg, conversion, P, "平日")
    oh2, rh2, _ = demand_hourly(survey, cfg, conversion, P, "休日")
    dem = {"平日": (oh, rh), "休日": (oh2, rh2)}
    ops_kw = dict(pattern="a", substitution=substitution, ebike_share=es)
    cal, specs, source = month_specs(cfg, year, month, dem, ops_kw, seed)
    L, ports, assign = optimal_layout(net, cfg, sh, k, es, station_cap=station_cap)
    ops = make_ops(cfg, weather="晴れ", **ops_kw)
    tg = tuple(sorted(set(CURVE_FAIL) | {round(1 - target, 4)}, reverse=True))
    res = min_fleet(L, ops, oh, rh, targets=tg, n_rep=len(specs) * passes, seed=seed, day_specs=specs)
    curve = pd.DataFrame([{"目標充足率": 1 - t, "合計": r["合計"], "自転車": r["自転車"], "キックボード": r["キックボード"],
                           "充足率": 1 - r["失敗率"]} for t, r in res.items() if not str(t).startswith("_") and r])
    curve = curve.sort_values("目標充足率").reset_index(drop=True)
    best = res.get(round(1 - target, 4))
    out = dict(conversion=conversion, year=year, month=month, station_cap=station_cap, substitution=substitution,
               k=k, target=target, source=source, n_days=len(cal), n_hol=int((cal["曜日区分"] == "休日").sum()),
               houses=net.houses, nodes=net.nodes, candidates=net.candidates, ports=list(ports), assign=assign,
               oh=oh, rh=rh, share=sh, curve=curve, best=best, n_houses=len(net.houses),
               districts=(net.houses["街区"].value_counts(sort=False).to_dict() if "街区" in net.houses else {}),
               dest={k_: cfg["destinations"][k_] for k_ in ("station", "aeon")})
    if best:
        F = fleet_for(L, best["自転車"], best["キックボード"], oh, rh, ops, seed=seed)
        ms, _ = month_eval(L, F, cfg, dem, month=month, seed=seed + 1, cal=cal, **ops_kw)
        out.update(init=F.init, walk_only=ms["全員徒歩の平均(分/片道)"], travel=ms["平均移動時間(分/片道)"])
    else:
        ms, _ = month_eval(L, fleet_for(L, 1, 0, oh, rh, ops), cfg, dem, month=month, seed=seed + 1, cal=cal, **ops_kw)
        out.update(init=None, walk_only=ms["全員徒歩の平均(分/片道)"], travel=None)
    return out


# ---------------------------------------------------------------- 描画の部品
def _frame(fig, title, cond, n):
    fig.patch.set_facecolor("white")
    fig.text(0.035, 0.915, title, fontsize=30, weight="bold", color=NAVY, va="center")
    fig.text(0.035, 0.85, cond, fontsize=15, color=MUTED, va="center")
    fig.add_artist(plt.Line2D([0.035, 0.965], [0.815, 0.815], color=LINE, lw=1.5))
    fig.text(0.965, 0.915, f"{n}/6", fontsize=14, color=MUTED, ha="right", va="center")


def _kpi(fig, slot, label, value, unit="", color=NAVY):
    y = (0.58, 0.36, 0.14)[slot]
    ax = fig.add_axes([0.65, y, 0.315, 0.19])
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.9, boxstyle="round,pad=0.01,rounding_size=0.06",
                                fc=BG, ec="none", transform=ax.transAxes))
    ax.text(0.08, 0.72, label, fontsize=15, color=MUTED, transform=ax.transAxes, va="center")
    ax.text(0.08, 0.32, value, fontsize=40 if len(str(value)) <= 6 else 32, weight="bold", color=color,
            transform=ax.transAxes, va="center")
    ax.text(0.92, 0.3, unit, fontsize=16, color=MUTED, transform=ax.transAxes, va="center", ha="right")


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#C9CED6")
    ax.tick_params(labelsize=13, colors=MUTED)
    ax.grid(axis="y", color="#EEF0F3")
    ax.set_axisbelow(True)


def _map(ax, c, show_ports):
    nd = c["nodes"]
    if nd is not None:
        ax.scatter(nd["lon"], nd["lat"], s=1.2, c="#DDE1E7", zorder=1)
    h = c["houses"]
    cols = [PORT_COLORS[a % len(PORT_COLORS)] if show_ports else "#8A93A6" for a in c["assign"]]
    ax.scatter(h["lon"], h["lat"], s=10, c=cols, alpha=.55, lw=0, zorder=2)
    if show_ports:
        cand = c["candidates"].iloc[c["ports"]]
        for i, (_, p) in enumerate(cand.iterrows()):
            col = PORT_COLORS[i % len(PORT_COLORS)]
            ax.scatter(p["lon"], p["lat"], s=420, marker="s", c=col, edgecolors="white", lw=2, zorder=4)
            ax.annotate(f"P{i + 1}", (p["lon"], p["lat"]), color="white", ha="center", va="center", fontsize=12,
                        weight="bold", zorder=5)
            if c["init"] is not None:
                ax.annotate(f"自{c['init'][i, 0]}・キ{c['init'][i, 1]}", (p["lon"], p["lat"]), xytext=(14, -18),
                            textcoords="offset points", fontsize=12, color=INK, zorder=5,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=.85))
    for key, d in c["dest"].items():
        p = d["port"]
        ax.scatter(p["lon"], p["lat"], s=320, marker="D", c=ORANGE, edgecolors="white", lw=2, zorder=4)
        ax.annotate(d["port_label"], (p["lon"], p["lat"]), xytext=(10, 8), textcoords="offset points",
                    fontsize=14, color=INK, weight="bold", zorder=5)
    lat = np.r_[h["lat"], [d["port"]["lat"] for d in c["dest"].values()]]
    lon = np.r_[h["lon"], [d["port"]["lon"] for d in c["dest"].values()]]
    ax.set_xlim(lon.min() - 0.0008, lon.max() + 0.0025)
    ax.set_ylim(lat.min() - 0.0006, lat.max() + 0.0006)
    ax.set_aspect(1 / np.cos(np.radians(lat.mean())))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(LINE)


def _cond(c):
    dist = "・".join(f"{k}{v}" for k, v in c["districts"].items())
    return (f"条件：{c['n_houses']}戸（{dist}）／転換率{c['conversion']:.0%}／{c['year']}年{c['month']}月"
            f"（平日{c['n_days'] - c['n_hol']}日・休日{c['n_hol']}日・{c['source']}）／"
            f"{'代替乗車あり' if c['substitution'] else '代替乗車なし'}／駅ポート{c['station_cap']}台")


def _fig():
    return plt.figure(figsize=(16, 9))


# ---------------------------------------------------------------- 6画面
def fig_area(c):
    fig = _fig()
    dist = "＋".join(f"{k}{v}戸" for k, v in c["districts"].items())
    labels = "・".join(d["port_label"] for d in c["dest"].values())
    _frame(fig, f"南栗橋の住宅{c['n_houses']}戸と、駅・商業施設を結ぶ", f"対象：{dist}／{labels}", 1)
    _map(fig.add_axes([0.035, 0.05, 0.58, 0.72]), c, show_ports=False)
    _kpi(fig, 0, "住宅", f"{c['n_houses']}", "戸")
    _kpi(fig, 1, "目的地まで（徒歩・平均）", f"{c['walk_only']:.0f}", "分")
    _kpi(fig, 2, "車種", "2", "種（自転車・キックボード）", TEAL)
    return fig


def fig_demand(c):
    fig = _fig()
    _frame(fig, "利用希望は朝の駅行きと夕方の帰宅に集中する", _cond(c), 2)
    ax = fig.add_axes([0.06, 0.1, 0.56, 0.66])
    _style(ax)
    hrs = np.arange(5, 23)
    out, ret = c["oh"].sum(1)[hrs], c["rh"].sum(1)[hrs]
    ax.bar(hrs - 0.2, out, 0.4, color=NAVY, label="往路（自宅→駅・イオン）")
    ax.bar(hrs + 0.2, ret, 0.4, color=TEAL, label="復路（駅・イオン→自宅）")
    ax.set_xticks(hrs)
    ax.set_xlabel("時刻", fontsize=14, color=MUTED)
    ax.set_ylabel("回／時（平日）", fontsize=14, color=MUTED)
    ax.legend(fontsize=14, frameon=False, loc="upper center")
    peak_h = int(hrs[np.argmax(out)])
    _kpi(fig, 0, "1日の利用希望（平日・晴れ）", f"{out.sum() + ret.sum():.0f}", "回")
    _kpi(fig, 1, f"朝のピーク（{peak_h}時台）", f"{out.max():.1f}", "回")
    _kpi(fig, 2, "駅への割合", f"{c['share'][0]:.0%}", "", TEAL)
    return fig


def fig_optimal(c):
    fig = _fig()
    b = c["best"]
    if not b:
        _frame(fig, f"充足率{c['target']:.0%}には届かない（探索範囲内）", _cond(c), 3)
        _map(fig.add_axes([0.035, 0.05, 0.58, 0.72]), c, show_ports=True)
        return fig
    ful = 1 - b["失敗率"]
    _frame(fig, f"住宅側{c['k']}か所・{b['合計']}台で、利用希望の{ful:.0%}に応えられる", _cond(c), 3)
    _map(fig.add_axes([0.035, 0.05, 0.58, 0.72]), c, show_ports=True)
    _kpi(fig, 0, "台数（自転車＋キックボード）", f"{b['自転車']}＋{b['キックボード']}", "台")
    _kpi(fig, 1, "充足率（1か月）", f"{ful:.0%}", "", TEAL)
    dt = (c["travel"] - c["walk_only"]) if c["travel"] is not None else 0
    _kpi(fig, 2, "移動時間（全員徒歩と比べて）", f"{dt:+.1f}", "分／片道")
    return fig


def fig_curve(c):
    fig = _fig()
    cv = c["curve"]
    pick = {}
    for t in (0.8, 0.95):
        row = cv[np.isclose(cv["目標充足率"], t)]
        pick[t] = int(row["合計"].iloc[0]) if len(row) else None
    title = "　".join(f"{t:.0%}に応えるには{n}台" for t, n in pick.items() if n) or "台数と充足率"
    _frame(fig, title.replace("　", "、", 1), _cond(c), 4)
    ax = fig.add_axes([0.06, 0.1, 0.56, 0.66])
    _style(ax)
    ax.plot(cv["合計"], cv["充足率"], color=NAVY, lw=4, marker="o", ms=8)
    for (t, n), col in zip(pick.items(), (TEAL, ORANGE)):
        if n:
            ax.axhline(t, color=col, ls="--", lw=2)
            ax.axvline(n, color=col, ls=":", lw=2)
            ax.annotate(f"{t:.0%}：{n}台", (n, t), xytext=(10, -28), textcoords="offset points", fontsize=15,
                        color=col, weight="bold")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, max(cv["合計"].max() * 1.08, 10) if len(cv) else 10)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("台数（自転車＋キックボード、配置・内訳はそれぞれ最適化）", fontsize=14, color=MUTED)
    ax.set_ylabel("充足率（1か月）", fontsize=14, color=MUTED)
    _kpi(fig, 0, "充足率80%", f"{pick[0.8] or '—'}", "台")
    _kpi(fig, 1, "充足率95%", f"{pick[0.95] or '—'}", "台", TEAL)
    small = cv.iloc[0] if len(cv) else None
    if small is not None:
        _kpi(fig, 2, f"{int(small['合計'])}台のとき", f"{small['充足率']:.0%}", "", ORANGE)
    return fig


def fig_pilot(pilot: pd.DataFrame, n_total: int, note: str):
    """pilot：転換率ごとの最良の組み合わせ（src.pilot の結果）。"""
    fig = _fig()
    cap = n_total * 2
    _frame(fig, f"{n_total}台の実証では、利用は1日約{cap}回で頭打ちになる", note, 5)
    ax = fig.add_axes([0.06, 0.1, 0.56, 0.66])
    _style(ax)
    x = (pilot["転換率"] * 100).round().astype(int)
    ax.bar(x, pilot["利用"], 3.6, color=NAVY, label="利用できた")
    ax.bar(x, pilot["断られた/日"], 3.6, bottom=pilot["利用"], color=GRAY, label="断られた")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}%" for v in x])
    ax.set_xlabel("転換率", fontsize=14, color=MUTED)
    ax.set_ylabel("回／日（平日・晴れ）", fontsize=14, color=MUTED)
    ax.legend(fontsize=14, frameon=False, loc="upper left")
    row = pilot[np.isclose(pilot["転換率"], 0.10)]
    _kpi(fig, 0, "利用の上限", f"約{cap}", "回／日")
    if len(row):
        _kpi(fig, 1, "転換率10%の充足率", f"{row['充足率'].iloc[0]:.0%}", "", TEAL)
    _kpi(fig, 2, "実証で記録すべきこと", "断られた数", "", ORANGE)
    return fig


def fig_sensitivity(sens: pd.DataFrame | None):
    fig = _fig()
    if sens is None or sens.empty:
        _frame(fig, "感度分析", "結果がありません（python scripts/run_all.py で作成）", 6)
        return fig
    s = sens[sens["運用"] == "a"].copy()
    s["達成"] = s["達成"].astype(bool)
    g = s[s["達成"]].groupby(["項目", "値"])["合計台数"].median()
    dem = g.loc["需要（転換率）"] if "需要（転換率）" in g.index.get_level_values(0) else None
    base = float(dem.iloc[len(dem) // 2]) if dem is not None else float(g.median())
    base_v = dem.index[len(dem) // 2] if dem is not None else None
    items = [i for i in ("需要（転換率）", "駅ポート容量", "自転車の比率", "歩行速度（km/h）", "手続き時間（秒）")
             if i in g.index.get_level_values(0)]
    rows = []
    for it in items:
        v = g.loc[it]
        rows.append((it, v.iloc[0] - base, v.iloc[-1] - base))
    rows.sort(key=lambda t: abs(t[1]) + abs(t[2]))
    order = [r[0] for r in rows[::-1]]
    name = {"需要（転換率）": "需要", "駅ポート容量": "駅ポート", "自転車の比率": "自転車の比率",
            "歩行速度（km/h）": "歩行速度", "手続き時間（秒）": "手続き時間"}
    _frame(fig, f"台数を最も左右するのは「{name[order[0]]}」と「{name[order[1]]}」",
           f"基準：転換率{base_v:.0%}ほか（各点の中央値、失敗率5%・運用a）" if base_v is not None else "", 6)
    ax = fig.add_axes([0.2, 0.1, 0.42, 0.66])
    _style(ax)
    ax.grid(axis="x", color="#EEF0F3")
    ax.grid(axis="y", visible=False)
    y = np.arange(len(rows))
    ax.barh(y, [t[1] for t in rows], color=GRAY, height=0.55, label="下限の値のとき")
    ax.barh(y, [t[2] for t in rows], color=NAVY, height=0.55, label="上限の値のとき")
    ax.set_yticks(y)
    ax.set_yticklabels([t[0] for t in rows], fontsize=15, color=INK)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel(f"最小台数の増減（台）　基準 {base:.0f}台", fontsize=14, color=MUTED)
    ax.legend(fontsize=13, frameon=False, loc="lower right")
    _kpi(fig, 0, "最も効く要因", name[order[0]], "", NAVY)
    _kpi(fig, 1, "次に効く要因", name[order[1]], "", TEAL)
    _kpi(fig, 2, "ほぼ効かない", name[order[-1]], "", MUTED)
    return fig


# ---------------------------------------------------------------- 出力
def to_png(fig, dpi=120) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


def to_pptx(pngs: list[bytes]) -> bytes:
    """画像を1枚ずつ全面に貼った16:9のPPT。"""
    from pptx import Presentation
    from pptx.util import Inches
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    for b in pngs:
        s = prs.slides.add_slide(prs.slide_layouts[6])
        s.shapes.add_picture(io.BytesIO(b), 0, 0, width=prs.slide_width, height=prs.slide_height)
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()
