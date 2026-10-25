"""実証（台数固定）ページ：10台などの限られた台数で、転換率ごとにどこまで利用希望に応えられるか。"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.loaders import ROOT
from src.pilot import pilot_best, pilot_grid, pilot_year
from src.ui import header_warnings, load_or_stop

BASE_CFG, SURVEY, NET, FLAGS, DATA_KEY = load_or_stop("pilot")
cfg = BASE_CFG

st.title("実証（台数固定）")
st.caption("台数を決めたとき（例：実証の10台）、住宅側ポートの数と自転車・キックボードの割合をどうすると、"
           "どれだけ利用希望に応えられるか。運用は夜1回の再配置、代替乗車は設定どおり。")
header_warnings(NET, FLAGS)

with st.sidebar:
    st.markdown("**実証の条件**")
    n_total = st.number_input("台数（自転車＋キックボード）", 1, 60, 10, key="pl_n")
    opts = [round(c * 100) for c in cfg["simulation"]["conversion_options"]]
    conv = st.select_slider("利用転換率", options=opts, value=10, format_func=lambda v: f"{v}%", key="pl_conv") / 100
    ks = st.multiselect("比べる住宅側ポート数", [1, 2, 3, 4], default=[1, 2, 3], key="pl_ks")
    cap = st.number_input("駅ポートの容量（台）", 1, 200, int(cfg["station_port"]["max_slots"]), key="pl_cap")
    n_rep = st.select_slider("シミュレーション日数", [60, 100, 200], value=100, key="pl_rep")
if not ks:
    st.warning("ポート数を1つ以上選んでください。")
    st.stop()


@st.cache_data(show_spinner="組み合わせを計算中…")
def grid_for(conv, n_total, ks, cap, n_rep, data_key):
    return pilot_grid(NET, cfg, SURVEY, conv, n_total=n_total, ks=ks, station_cap=cap, n_rep=n_rep)


@st.cache_data(show_spinner="1年分を計算中…")
def year_for(conv, ports, nb, nk, cap, data_key):
    b = pd.Series({"ポート数": ports, "自転車": nb, "キックボード": nk})
    return pilot_year(NET, cfg, SURVEY, conv, b, station_cap=cap)


g = grid_for(conv, int(n_total), tuple(sorted(ks)), int(cap), int(n_rep), DATA_KEY)
best = pilot_best(g)
ys, bm, cost = year_for(conv, int(best["ポート数"]), int(best["自転車"]), int(best["キックボード"]), int(cap), DATA_KEY)

st.subheader(f"最も応えられる組み合わせ（転換率{conv:.0%}・{n_total}台）")
c1, c2, c3, c4 = st.columns(4)
c1.metric("住宅側ポート", f"{int(best['ポート数'])}か所")
c2.metric("内訳", f"自{int(best['自転車'])}・キ{int(best['キックボード'])}", help=f"駅ポートの朝の初期配置 {int(best['駅初期台数'])}台")
c3.metric("充足率（平日・晴れ）", f"{best['充足率']:.0%}", help=f"1年平均 {ys['充足率']:.0%}（休日・雨の日を含む）")
c4.metric("断られる回数（平日・晴れ）", f"{best['断られた/日']:.1f}回/日",
          help=f"利用希望 {best['利用希望']:.1f}回/日のうち、利用 {best['利用']:.1f}回/日")
st.caption(f"1年：利用 {cost['年間利用回数']:,.0f}回・年間費用 {cost['年間総費用'] / 1e4:,.0f}万円"
           f"（1回あたり {cost['1トリップあたり費用']:,.0f}円）・稼働率 {cost['稼働率(回/台/日)']:.2f}回/台/日")
if best["利用"] >= n_total * 1.8:
    st.info(f"利用は1日 約{n_total * 2}回（{n_total}台×朝夕1往復）で頭打ちです。これを超える利用希望は断られるため、"
            "実証では「借りられなかった操作」を記録しないと本当の需要が分かりません。")

st.subheader("ポート数 × 自転車の台数ごとの充足率（平日・晴れ）")
hm = g.pivot_table(index="ポート数", columns="自転車", values="充足率")
fig = px.imshow(hm, text_auto=".0%", aspect="auto", color_continuous_scale="Blues", zmin=0, zmax=1,
                labels=dict(x="自転車の台数（残りはキックボード）", y="住宅側ポート数", color="充足率"))
fig.update_layout(height=60 + 70 * len(hm), margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("月別の充足率（1年）")
fig2 = px.bar(bm, x="月", y="充足率", text_auto=".0%")
fig2.update_layout(height=300, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(dtick=1))
st.plotly_chart(fig2, use_container_width=True)

pf = ROOT / "data" / "output" / "pilot" / f"pilot_{int(n_total)}_best.csv"
if pf.exists():
    st.subheader("転換率ごとの見込み（scripts/run_pilot.py の結果）")
    b = pd.read_csv(pf)
    long = b.melt(id_vars="転換率", value_vars=["利用希望", "利用"], var_name="", value_name="回/日")
    long["転換率"] = (long["転換率"] * 100).round().astype(int).astype(str) + "%"
    fig3 = px.bar(long, x="転換率", y="回/日", color="", barmode="overlay", opacity=0.85)
    fig3.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)
    t = b[["転換率", "ポート数", "自転車", "キックボード", "充足率", "利用希望", "利用", "断られた/日", "年_充足率", "年間総費用"]].copy()
    t["転換率"] = t["転換率"].map(lambda v: f"{v:.0%}")
    for c in ("充足率", "年_充足率"):
        t[c] = t[c].map(lambda v: f"{v:.0%}")
    t["年間総費用"] = (t["年間総費用"] / 1e4).round().astype(int).astype(str) + "万円"
    st.dataframe(t.round(1), hide_index=True, use_container_width=True)

with st.expander("全組み合わせの表"):
    st.dataframe(g.round(3), hide_index=True, use_container_width=True)
