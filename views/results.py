"""計算結果ページ：全条件×10回（run_study.py）と感度分析（run_sensitivity.py）の結果を条件で絞って見る。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.loaders import ROOT

OUT = ROOT / "data" / "output"
METRICS = {
    "合計台数": ("合計台数", "台", "{:.0f}"),
    "自転車": ("自転車", "台", "{:.0f}"),
    "キックボード": ("キックボード", "台", "{:.0f}"),
    "年間総費用（万円）": ("費用_年間総費用", "万円", "{:,.0f}"),
    "1トリップあたり費用（円）": ("費用_1トリップあたり費用", "円", "{:,.0f}"),
    "稼働率（回/台/日）": ("費用_稼働率(回/台/日)", "", "{:.2f}"),
    "充足率（代表月）": ("月_充足率", "", "{:.1%}"),
    "返却失敗率（代表月）": ("月_返却失敗率", "", "{:.1%}"),
    "代替乗車率（代表月）": ("月_代替乗車率", "", "{:.1%}"),
    "充足率（1年）": ("年_充足率", "", "{:.1%}"),
}
DIMS = ["需要シナリオ", "代替乗車", "運用", "駅ポート容量", "目標失敗率"]

st.title("計算結果")
st.caption("`python scripts/run_all.py`（または run_study.py・run_sensitivity.py）で作った結果を表示します。"
           "各条件を乱数を変えて繰り返した結果のばらつきも見られます。")

sets = sorted([p.parent for p in OUT.glob("*/results.csv")], key=lambda p: p.name)
if not sets:
    st.info("結果がありません。`python scripts/run_all.py` を実行してください。")
    st.stop()


def _label(p):
    meta = p / "meta.json"
    d = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
    dist = "・".join(d.get("districts") or []) or "全街区"
    return f"{p.name}（{dist}）"


with st.sidebar:
    folder = st.selectbox("結果のフォルダ", sets, format_func=_label, index=[p.name for p in sets].index("study")
                          if "study" in [p.name for p in sets] else 0)


@st.cache_data
def load(path: str, mtime: float):
    d = pd.read_csv(path)
    d["達成"] = d["達成"].astype(bool)
    if "費用_年間総費用" in d:
        d["費用_年間総費用"] = d["費用_年間総費用"] / 1e4
    return d


p = folder / "results.csv"
d = load(str(p), p.stat().st_mtime)
with st.sidebar:
    st.markdown("**条件で絞る**")
    sel = {}
    for c in DIMS:
        opts = sorted(d[c].dropna().unique().tolist(), key=lambda v: (str(type(v)), v))
        default = opts
        if c == "目標失敗率" and 0.05 in opts:
            default = [0.05]
        sel[c] = st.multiselect(c, opts, default=default, key=f"res_{c}",
                                format_func=(lambda v: f"{v:.0%}") if c == "目標失敗率" else str)
x = d.copy()
for c, v in sel.items():
    x = x[x[c].isin(v)]
if x.empty:
    st.warning("条件に合う結果がありません。")
    st.stop()

# ---------- 条件ごとの表 ----------
st.subheader("条件ごとの結果（中央値と最小–最大）")
mname = st.selectbox("指標", [m for m, (col, _, _) in METRICS.items() if col in x], index=0)
col, unit, fmt = METRICS[mname]
keys = [c for c in DIMS if len(sel[c]) > 1] or ["需要シナリオ"]
g = x.groupby(keys)
tab = g.agg(回数=("回", "nunique"), 達成=("達成", "sum"))
ok = x[x["達成"]].groupby(keys)[col]
tab["中央値"] = ok.median()
tab["最小"] = ok.min()
tab["最大"] = ok.max()
tab = tab.reset_index()
show = tab.copy()
for c in ("中央値", "最小", "最大"):
    show[c] = show[c].map(lambda v: "未達" if pd.isna(v) else fmt.format(v))
show["達成"] = show.apply(lambda r: f"{r['達成']}/{r['回数']}回", axis=1)
st.dataframe(show.drop(columns="回数"), hide_index=True, use_container_width=True)
st.download_button("この表をCSVで保存", tab.to_csv(index=False).encode("utf-8-sig"), file_name=f"結果_{mname}.csv",
                   mime="text/csv")

# ---------- ばらつき ----------
st.subheader("ばらつき（箱ひげ）")
c1, c2 = st.columns(2)
xa = c1.selectbox("横軸", DIMS, index=0)
ca = c2.selectbox("色分け", ["なし"] + [c for c in DIMS if c != xa], index=1 + [c for c in DIMS if c != xa].index("代替乗車")
                  if "代替乗車" != xa else 0)
fig = px.box(x[x["達成"]], x=xa, y=col, color=None if ca == "なし" else ca, points="all",
             labels={col: f"{mname}"})
fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), legend_title_text=ca if ca != "なし" else "")
st.plotly_chart(fig, use_container_width=True)

# ---------- 季節 ----------
mcols = [f"月別利用_{m}" for m in range(1, 13)]
if all(c in x for c in mcols):
    st.subheader("月別の1日あたり利用回数（1年の計算、中央値）")
    keys2 = [c for c in ("需要シナリオ", "代替乗車", "駅ポート容量") if len(sel[c]) > 1] or ["需要シナリオ"]
    mm = x[x["達成"]].groupby(keys2)[mcols].median().reset_index()
    mm["条件"] = mm[keys2].astype(str).agg("・".join, axis=1)
    long = mm.melt(id_vars="条件", value_vars=mcols, var_name="月", value_name="利用（回/日）")
    long["月"] = long["月"].str.replace("月別利用_", "").astype(int)
    fig2 = px.line(long, x="月", y="利用（回/日）", color="条件", markers=True)
    fig2.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(dtick=1))
    st.plotly_chart(fig2, use_container_width=True)

# ---------- 感度分析・追加検証 ----------
sp = folder / "sensitivity.csv"
if sp.exists():
    st.subheader("感度分析（各点の中央値）")
    s = pd.read_csv(sp)
    s["達成"] = s["達成"].astype(bool)
    gg = s.groupby(["項目", "値", "運用"])
    st_ = gg[["合計台数", "年間総費用"]].median()
    st_["達成"] = gg["達成"].mean().map(lambda v: f"{v:.0%}")
    st_["年間総費用"] = (st_["年間総費用"] / 1e4).round(0)
    item = st.selectbox("項目", sorted(s["項目"].unique()))
    st.dataframe(st_.loc[item].reset_index().rename(columns={"年間総費用": "年間総費用（万円）"}), hide_index=True,
                 use_container_width=True)
for name, title in (("full_return_test.csv", "満車時の扱い"), ("discourage_test.csv", "利用離れの強さ α"),
                    ("rebalance_buffer_test.csv", "リバランスbで駅に残す台数")):
    fp = folder / name
    if fp.exists():
        with st.expander(f"追加検証：{title}"):
            st.dataframe(pd.read_csv(fp), hide_index=True, use_container_width=True)
