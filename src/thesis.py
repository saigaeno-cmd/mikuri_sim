"""修士論文用：比較ケースの計算と、1ケース1ページの結果シート・比較の表・曲線（16:9 の画像）。

ケースは data/conditions/thesis_cases.yaml。計算は src.opt_multi（①・②と同じモデル）。
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import opt_multi as om  # noqa: E402
from . import timesave as ts  # noqa: E402
from .engine import TYPES, fleet_label  # noqa: E402
from .sheet import GRAY, INK, LINE, MUTED, NAVY, ORANGE, PORT_COLORS, TEAL, _box, _hourly, _table  # noqa: E402

DEST_COLOR = {"station": NAVY, "aeon": ORANGE, "park": "#2E7D32", "anytime": "#7B1FA2"}


# ---------------------------------------------------------------- 計算
def case_params(base: dict, case: dict) -> dict:
    p = {**base, **{k: v for k, v in case.items() if k not in ("id", "group", "name")}}
    return p


def run_case(net, cfg, survey, p: dict, seed: int = 0) -> dict:
    """1ケース1回（乱数 seed）。"""
    if p["model"] == "time":
        cfg = {**cfg, "opt_time": {**cfg["opt_time"], "port_cap": p["port_cap"]}}
        return om.compute(net, cfg, survey, dests=p["dests"], vehicles=p["vehicles"], target=p["target"],
                          k_max=int(p["k_max"]), A=float(p["A"]), X=float(p["X"]), seed=seed)
    cands = net.candidates[["lat", "lon"]].assign(name=[f"候補{i}" for i in net.candidates["cand_id"]],
                                                  cap=int(p["port_cap"]))
    return om.evaluate_candidates(net, cfg, survey, cands, dests=p["dests"], vehicles=p["vehicles"], target=p["target"],
                                  k_max=int(p["k_max"]), conversion=float(p["conversion"]),
                                  dest_caps={k: float(v) for k, v in p["dest_caps"].items()}, seed=seed)


def summary(R: dict) -> dict:
    """比較の表に載せる値。"""
    if R.get("error"):
        return {"エラー": R["error"]}
    sel, ref, ms = R.get("sel"), R.get("ref"), R["ms"]
    fl = sel or ref
    return {"達成": sel is not None, "ポート数": R["L"].k, "合計台数": fl["合計"], "自転車": fl["自転車"],
            "キックボード": fl["キックボード"], "4輪": fl.get("4輪", 0), "充足率": ms["充足率"],
            "最大充足率": R.get("best_possible", np.nan), "駅ポート必要容量": R["station_need"],
            "平均短縮移動時間": R["saving"], "利用/日": ms["利用"], "利用希望/日": ms["利用希望"],
            "年間総費用(万円)": R["cost"]["年間総費用"] / 1e4, "1回あたり費用(円)": R["cost"]["1トリップあたり費用"],
            "対象の住宅": R.get("n_hh", np.nan)}


def heatmap(net, cfg, p: dict) -> pd.DataFrame:
    """①の需要発生条件（徒歩 A 分以内 × 短縮 X 分以上）ごとの対象住宅数（ポート k か所を選び直す）。"""
    cfg2, oc, purposes, on, kw = om.prepare(cfg, p["dests"], p["vehicles"])
    wo, mob, cands, dsp, egress, hd, Wc = om.time_tensors(net, cfg2, p["dests"], purposes, oc["procedure_sec"] / 60)
    acc = ts.access_time(net, cfg2)
    one = np.ones(len(hd))
    mls = {a: ts.limit_access(mob, acc, a) for a in p["A"]}
    sols = [ts.solve_max_cover(wo, mls[a], int(p["k_max"]), x, one) for a in p["A"] for x in p["X"]]
    rows = []
    for a in p["A"]:
        for x in p["X"]:
            # 近似解のずれを避けるため、どの条件で見つけたポートの組も試し、最も多いものを使う
            best = max((ts.coverage(wo, mls[a], S, x, one)[0] for S in sols), key=lambda q: (q.any(1).sum(), q.sum()))
            rows.append({"徒歩（分以内）": a, "短縮（分以上）": x, "対象の住宅": int(best.any(1).sum()),
                         "往復/日": int(best.sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 描画
def _fmt_range(vals, fmt="{:.0f}", pct=False):
    v = np.asarray([x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))], float)
    if not len(v):
        return "—"
    f = (lambda z: f"{z:.0%}") if pct else (lambda z: fmt.format(z))
    med = np.median(v)
    return f(med) if v.min() == v.max() else f"{f(med)}（{f(v.min())}–{f(v.max())}）"


def _map(ax, R, cfg, roads):
    L, init = R["L"], R["init"]
    if roads is not None:
        ax.plot(roads[0], roads[1], color="#D5DAE1", lw=0.8, zorder=1)
    a = np.asarray(R["assign"])
    hh = R["_houses"]
    cols = [PORT_COLORS[x % len(PORT_COLORS)] if x >= 0 else "#B9BFC9" for x in a]
    ax.scatter(hh["lon"], hh["lat"], s=14, c=cols, alpha=[0.9 if x >= 0 else 0.45 for x in a], lw=0, zorder=2)
    cs = R["cands"].iloc[R["S"]]
    for j, (_, p) in enumerate(cs.iterrows()):
        col = PORT_COLORS[j % len(PORT_COLORS)]
        ax.scatter(p["lon"], p["lat"], s=330, marker="s", c=col, edgecolors="white", lw=2, zorder=4)
        ax.annotate(f"P{j + 1}", (p["lon"], p["lat"]), color="white", ha="center", va="center", fontsize=10,
                    weight="bold", zorder=5)
        ax.annotate(f"{fleet_label(init[j])}/収納{int(L.caps[j])}", (p["lon"], p["lat"]), xytext=(11, -15),
                    textcoords="offset points", fontsize=9, color=INK, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=.9))
    pts = [hh[["lon", "lat"]].to_numpy()]
    oc = cfg["opt_time"]
    rows = {"station": L.k, "aeon": L.k + 1}
    extra = [d for d in R["dests"] if d not in rows]
    for i, d in enumerate(extra):
        rows[d] = L.k + 2 + i
    for d in ("station", "aeon", "park", "anytime"):
        if d not in R["dests"] and d not in ("station", "aeon"):
            continue
        p = cfg["destinations"][d].get("port") or oc["dest_ports"][d]
        n = int(init[rows[d]].sum()) if d in rows else 0
        cap = L.caps[rows[d]] if d in rows else np.inf
        ax.scatter(p["lon"], p["lat"], s=230, marker="D", c=DEST_COLOR[d], edgecolors="white", lw=2, zorder=4)
        ax.annotate(f"{om.DEST_LABEL[d]}（初期{n}台・収納{'上限なし' if not np.isfinite(cap) or cap >= 900 else int(cap)}）",
                    (p["lon"], p["lat"]), xytext=(8, 6), textcoords="offset points", fontsize=9.5, color=INK,
                    weight="bold", zorder=5, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=.85))
        pts.append([[p["lon"], p["lat"]]])
    pts = np.vstack(pts)
    lat = pts[:, 1]
    ax.set_xlim(pts[:, 0].min() - 0.0006, pts[:, 0].max() + 0.0022)
    ax.set_ylim(lat.min() - 0.0005, lat.max() + 0.0006)
    ax.set_aspect(1 / np.cos(np.radians(lat.mean())))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(LINE)


def case_sheet(R: dict, cfg, title: str, cond: str, stats: dict | None, roads=None):
    """1ケース1ページ（①②と同じ並び）。stats：複数回の中央値と幅の文字列。"""
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    fig.text(0.03, 0.955, title, fontsize=22, weight="bold", color=INK, va="center")
    fig.text(0.03, 0.905, cond, fontsize=9.5, color=MUTED, va="center", linespacing=1.5)
    sel, ref, ms, L = R.get("sel"), R.get("ref"), R["ms"], R["L"]
    fl = sel or ref
    st = stats or {}
    note = "" if sel else "（目標未達・容量まで埋めた参考値）"
    fig.text(0.03, 0.85, "最適ポート配置" + note, fontsize=16, weight="bold", color=INK if sel else "#B42318",
             va="center")
    w, gap, x0 = 0.226, 0.0127, 0.03
    k1 = [("ポート数", f"{L.k} か所", f"対象の住宅 {R.get('n_hh', '—')}/{R.get('n_houses', '—')}戸"),
          ("駅ポート必要容量", f"{R['station_need']} 台", st.get("駅ポート必要容量", "")),
          ("充足率", f"{ms['充足率']:.0%}", (st.get("充足率", "") + "　" if st.get("充足率") else "")
           + f"目標{R['target']:.0%}"),
          ("平均短縮移動時間", f"{R['saving']:.1f} 分", st.get("平均短縮移動時間", ""))]
    for i, (lab, val, sub) in enumerate(k1):
        _box(fig, [x0 + i * (w + gap), 0.705, w, 0.12], lab, val, sub)
    ax = fig.add_axes([0.03, 0.565, 0.94, 0.125])
    ax.axis("off")
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((0.003, 0.04), 0.994, 0.92, boxstyle="round,pad=0.0,rounding_size=0.12",
                                fc="white", ec=INK, lw=1.6, transform=ax.transAxes))
    k2 = [("合計車両台数", f"{fl['合計']} 台", st.get("合計台数", "")),
          ("電動アシスト自転車", f"{fl['自転車']} 台", st.get("自転車", "")),
          ("電動キックボード", f"{fl['キックボード']} 台", st.get("キックボード", "")),
          ("4輪モビリティ", f"{fl.get('4輪', 0)} 台", st.get("4輪", ""))]
    for i, (lab, val, sub) in enumerate(k2):
        cx = 0.125 + i * 0.25
        ax.text(cx, 0.72, lab, fontsize=15, ha="center", va="center", transform=ax.transAxes, color=INK)
        ax.text(cx, 0.36, val, fontsize=24, weight="bold", ha="center", va="center", transform=ax.transAxes,
                color=NAVY)
        if sub:
            ax.text(cx, 0.12, sub, fontsize=9.5, ha="center", va="center", transform=ax.transAxes, color=MUTED)
        if i:
            ax.plot([i * 0.25, i * 0.25], [0.15, 0.85], color=MUTED, lw=1, transform=ax.transAxes)
    fig.text(0.03, 0.535, "ポート場所", fontsize=15, weight="bold", color=INK, va="center")
    _map(fig.add_axes([0.03, 0.03, 0.55, 0.49]), R, cfg, roads)
    fig.text(0.61, 0.535, "時間帯別の利用希望と充足", fontsize=12, weight="bold", color=INK, va="center")
    _hourly(fig.add_axes([0.645, 0.32, 0.33, 0.19]), ms["_hourly"])
    fig.text(0.61, 0.265, "ポートごと（台数は初期配置）", fontsize=12, weight="bold", color=INK, va="center")
    peaks = ms["ポート別最大駐車(95%点)"]
    init = R["init"]
    names = [f"P{j + 1}" for j in range(L.k)] + ["駅", "イオン"] + \
        [om.DEST_LABEL[d] for d in R["dests"] if d not in ("station", "aeon")]
    tb = pd.DataFrame({"ポート": names, "自転車": init[:, 0], "キック": init[:, 1], "4輪": init[:, 2],
                       "収納": [("なし" if not np.isfinite(c) or c >= 900 else f"{int(c)}") for c in L.caps],
                       "最大駐車": [int(x) for x in peaks[:len(names)]]})
    t = _table(fig.add_axes([0.61, 0.03, 0.365, 0.22]), tb)
    for (r_, c_), cell in t.get_celld().items():
        cell.set_fontsize(8.5)
    fig.text(0.03, 0.012, "色つきの点＝住宅（色は使うポート）、灰色＝利用しない住宅。ラベル：自＝電動アシスト自転車、"
             "キ＝電動キックボード、4輪＝4輪モビリティの初期台数／収納", fontsize=8, color=MUTED)
    return fig


def _slide(title, sub=""):
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    fig.text(0.035, 0.94, title, fontsize=26, weight="bold", color=INK, va="center")
    if sub:
        fig.text(0.035, 0.885, sub, fontsize=12, color=MUTED, va="center")
    fig.add_artist(plt.Line2D([0.035, 0.965], [0.86, 0.86], color=LINE, lw=1.5))
    return fig


def table_slide(title, df: pd.DataFrame, sub="", col_w=None, fs=11, highlight=None):
    fig = _slide(title, sub)
    ax = fig.add_axes([0.035, 0.04, 0.93, 0.8])
    ax.axis("off")
    t = ax.table(cellText=df.astype(str).values, colLabels=list(df.columns), loc="upper center", cellLoc="center",
                 colLoc="center", bbox=[0, max(0.0, 1 - 0.07 * (len(df) + 1)), 1, min(1.0, 0.07 * (len(df) + 1))])
    t.auto_set_font_size(False)
    t.set_fontsize(fs)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#DDE1E7")
        if col_w:
            cell.set_width(col_w[c])
        if r == 0:
            cell.set_facecolor("#F3F5F8")
            cell.set_text_props(color=MUTED, weight="bold")
        elif highlight is not None and highlight(r - 1):
            cell.set_facecolor("#FFF6E5")
    return fig


def curve_slide(curves: dict, target: float, sub=""):
    """台数と充足率の曲線（低位・高位など）。curves：{名前: DataFrame(合計台数, 充足率)}。"""
    fig = _slide("台数と充足率", sub)
    ax = fig.add_axes([0.07, 0.1, 0.62, 0.72])
    for (name, df), col in zip(curves.items(), (NAVY, ORANGE, TEAL, GRAY)):
        ax.plot(df["合計台数"], df["充足率"], marker="o", lw=3, ms=7, color=col, label=name)
        hit = df[df["充足率"] >= target]
        if len(hit):
            n = int(hit["合計台数"].iloc[0])
            ax.axvline(n, color=col, ls=":", lw=1.5)
            ax.annotate(f"{name}：{n}台で{target:.0%}", (n, target), xytext=(8, -20 - 18 * list(curves).index(name)),
                        textcoords="offset points", fontsize=12, color=col, weight="bold")
    ax.axhline(target, color=MUTED, ls="--", lw=1.5)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("合計台数（車種の比率は各ケースの内訳に合わせる）", fontsize=12, color=MUTED)
    ax.set_ylabel("充足率（9〜11月の3か月）", fontsize=12, color=MUTED)
    ax.grid(color="#EEF0F3")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=12, frameon=False, loc="lower right")
    tx = fig.add_axes([0.72, 0.1, 0.25, 0.72])
    tx.axis("off")
    lines = ["読み取り方"]
    for name, df in curves.items():
        mx = df["充足率"].max()
        lines.append(f"・{name}：最大 {mx:.0%}（{int(df.loc[df['充足率'].idxmax(), '合計台数'])}台）")
    lines += ["", "台数を増やしても充足率が頭打ちになる点は、", "ポートの容量（区画）や駅ポートの満車が", "上限を決めていることを示す。"]
    tx.text(0, 1, "\n".join(lines), fontsize=12.5, color=INK, va="top", linespacing=1.7)
    return fig


def ports_slide(df: pd.DataFrame, target: float, sub=""):
    """B：ポート数ごとの最大充足率と必要台数。df：ポート数, 最大充足率, 合計台数, 達成。"""
    fig = _slide("ポート数と充足率・台数", sub)
    ax = fig.add_axes([0.07, 0.1, 0.62, 0.72])
    ax.bar(df["ポート数上限"], df["合計台数"], color=["#C9D3EA" if not a else "#7C93C8" for a in df["達成"]], width=0.6,
           label="台数（濃い色＝目標達成（半数以上の回）の最小台数、薄い色＝容量まで埋めた参考）")
    ax.set_xlabel("ポート数上限（か所）", fontsize=12, color=MUTED)
    ax.set_ylabel("台数", fontsize=12, color=MUTED)
    ax.set_ylim(0, float(df["合計台数"].max()) * 1.35)
    ax.spines[["top"]].set_visible(False)
    a2 = ax.twinx()
    a2.plot(df["ポート数上限"], df["最大充足率"], color=ORANGE, marker="o", lw=3, ms=8, label="最大充足率（容量まで埋めたとき）")
    a2.axhline(target, color=MUTED, ls="--", lw=1.5)
    a2.set_ylim(0, 1.02)
    a2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    a2.spines[["top"]].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=11, frameon=False, loc="upper left")
    tx = fig.add_axes([0.73, 0.1, 0.24, 0.72])
    tx.axis("off")
    hit = df[df["達成"]]
    msg = (f"目標{target:.0%}を満たす最少のポート数：\n{int(hit['ポート数上限'].iloc[0])}か所（{int(hit['合計台数'].iloc[0])}台、"
           f"{hit['達成回'].iloc[0]}回で達成）" if len(hit) else f"ポート{int(df['ポート数上限'].max())}か所までは目標{target:.0%}に届かない")
    for x, y, r in zip(df["ポート数上限"], df["合計台数"], df.get("達成回", [""] * len(df))):
        ax.annotate(f"{y:.0f}台\n({r}回達成)" if r else f"{y:.0f}台", (x, y), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=10, color=INK)
    tx.text(0, 1, "読み取り方\n\n" + msg + "\n\nポートを増やすと住宅から近くなり、\n容量（区画）の合計も増える。",
            fontsize=12.5, color=INK, va="top", linespacing=1.7)
    return fig


def heatmap_slide(df: pd.DataFrame, sub=""):
    fig = _slide("需要発生条件と対象の住宅（①時間短縮型）", sub)
    pv = df.pivot(index="徒歩（分以内）", columns="短縮（分以上）", values="対象の住宅")
    ax = fig.add_axes([0.08, 0.1, 0.6, 0.72])
    im = ax.imshow(pv.values, cmap="Blues", aspect="auto", origin="lower")
    ax.set_xticks(range(pv.shape[1]), [f"{c:g}分" for c in pv.columns])
    ax.set_yticks(range(pv.shape[0]), [f"{r:g}分" for r in pv.index])
    ax.set_xlabel("短くなる時間（分以上）", fontsize=12, color=MUTED)
    ax.set_ylabel("住宅からポートまでの徒歩（分以内）", fontsize=12, color=MUTED)
    vmax = pv.values.max()
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            ax.text(j, i, f"{v}", ha="center", va="center", fontsize=13,
                    color="white" if v > vmax * 0.55 else INK, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.04, label="対象の住宅（戸）")
    tx = fig.add_axes([0.74, 0.1, 0.23, 0.72])
    tx.axis("off")
    tx.text(0, 1, "読み取り方\n\n条件をゆるめる（徒歩を長く・短縮を\n小さく）ほど、対象の住宅が増える。\n"
            "\n数字＝ポート3か所を選び直したときの\n対象の住宅（全230戸中）",
            fontsize=12.5, color=INK, va="top", linespacing=1.7)
    return fig
