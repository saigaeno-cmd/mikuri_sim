"""② ポート配置評価：地図上でポートを選び、車種別の台数と収納可能台数を指定して、充足率・合計移動時間・費用を算出する。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src import viz
from src.custom_ports import ACCEPT, parse_ports, snap_ports
from src.engine import EB, KB, Fleet, allocate_fleet
from src.export import save_ports
from src.loaders import ROOT
from src.plan import custom_layout, demand_hourly, make_ops, optimal_layout
from src.study import annual_cost, month_eval
from src.ui import demand_sidebar, fresh_cfg, header_warnings, load_or_stop, month_specs, ops_sidebar
from src.weather_data import weather_path

BASE_CFG, SURVEY, NET, FLAGS, DATA_KEY = load_or_stop("eval")
cfg = fresh_cfg(BASE_CFG)
COLS = ["name", "lat", "lon", "自転車", "キックボード", "収納"]
MAX_SNAP_M = 300
DEFAULT_CAP = int(BASE_CFG["service"].get("default_port_capacity", 10))

with st.sidebar:
    st.header("条件")
    st.subheader("需要")
    spec = demand_sidebar(cfg, SURVEY, key_prefix="ev_", show_day=False)
    ops_spec = ops_sidebar(cfg, key_prefix="ev_", show_period=False)

st.title("② ポート配置評価")
st.caption("地図をクリックして住宅側ポートを置き、表で車種別の台数と収納可能台数を指定すると、充足率・移動時間・年間費用を計算します。"
           "クリック位置は最寄りの道路上の点に合わせます。")
header_warnings(NET, FLAGS)

P, conv = spec.purposes, spec.conversion
oh_wd, rh_wd, sh = demand_hourly(SURVEY, cfg, conv, P, "平日", None, spec.station_share)
oh_hd, rh_hd, _ = demand_hourly(SURVEY, cfg, conv, P, "休日", None, spec.station_share)
oh, rh = oh_wd, rh_wd                          # 初期配置の自動配分は平日の需要で
DEM = {"平日": (oh_wd, rh_wd), "休日": (oh_hd, rh_hd)}
OPS_KW = dict(pattern=ops_spec.pattern, substitution=ops_spec.substitution, ebike_share=ops_spec.ebike_share,
              procedure_sec=ops_spec.procedure_sec)
ops = make_ops(cfg, weather="晴れ", **OPS_KW)
CAL, _, SOURCE = month_specs(cfg, ops_spec.year, ops_spec.month, DEM, OPS_KW)

# ---------- 配置の状態（表の初期値＋版番号。版が変わると表を作り直す） ----------
ss = st.session_state
if "ev2_ports" not in ss:
    ss.ev2_ports = pd.DataFrame(columns=COLS).astype({"name": str, "lat": float, "lon": float, "自転車": int,
                                                      "キックボード": int, "収納": int})
    ss.ev2_ver = 0
    ss.ev2_last_click = None
    ss.ev2_station = (0, 0)


def set_ports(df: pd.DataFrame, station=None):
    df = df.copy()
    for c in ("自転車", "キックボード"):
        if c not in df:
            df[c] = 0
    if "収納" not in df:
        df["収納"] = np.maximum(df["自転車"] + df["キックボード"], int(ss.get("ev2_newcap", DEFAULT_CAP)))
    ss.ev2_ports = df[COLS].reset_index(drop=True)
    if station is not None:
        ss.ev2_station = station
    ss.ev2_ver += 1
    st.rerun()


def next_name(df) -> str:
    used = set(df["name"])
    i = 1
    while f"地点{i}" in used:
        i += 1
    return f"地点{i}"


# 操作ボタン
b1, b2, b3, b4 = st.columns([1.2, 1.4, 1.4, 0.8])
with b1:
    k_load = st.number_input("最適配置のポート数", 1, cfg["simulation"]["max_ports"], 3, key="ev2_kload")
with b2:
    st.write("")
    if st.button("最適配置を読み込む", use_container_width=True,
                 help="移動時間最短の位置。台数は下の「合計台数」を自動配分（駅ポートを含む）"):
        L0, ports0, _ = optimal_layout(NET, cfg, sh, int(k_load), ops_spec.ebike_share, station_cap=ops_spec.station_cap)
        F0 = allocate_fleet(L0, ops, int(ss.get("ev2_nb", 15)), int(ss.get("ev2_nk", 10)), oh, rh)
        c = NET.candidates.loc[list(ports0)]
        cap0 = np.maximum(F0.init[:L0.k].sum(1), int(ss.get("ev2_newcap", DEFAULT_CAP)))
        set_ports(pd.DataFrame({"name": [f"P{j + 1}" for j in range(len(c))], "lat": c["lat"].to_numpy(),
                                "lon": c["lon"].to_numpy(), "自転車": F0.init[:L0.k, EB], "キックボード": F0.init[:L0.k, KB],
                                "収納": cap0}), station=(int(F0.init[L0.k, EB]), int(F0.init[L0.k, KB])))
with b3:
    up = st.file_uploader("ファイルから読み込む", type=ACCEPT, key="ev2_upload", label_visibility="collapsed")
    if up is not None and ss.get("ev2_upload_done") != (up.name, up.size):
        try:
            p = parse_ports(up.name, up.getvalue())
            ss.ev2_upload_done = (up.name, up.size)
            set_ports(p.assign(自転車=int(ss.get("ev2_newb", 3)), キックボード=int(ss.get("ev2_newk", 2)),
                               収納=int(ss.get("ev2_newcap", DEFAULT_CAP))))
        except Exception as e:  # noqa: BLE001
            st.error(f"{up.name} を読み込めませんでした：{e}")
with b4:
    st.write("")
    if st.button("全削除", use_container_width=True):
        set_ports(ss.ev2_ports.iloc[0:0], station=(0, 0))

# ポート表
edited = st.data_editor(
    ss.ev2_ports, key=f"ev2_editor_{ss.ev2_ver}", num_rows="dynamic", hide_index=True, use_container_width=True,
    column_config={"name": st.column_config.TextColumn("名称"),
                   "lat": st.column_config.NumberColumn("緯度", format="%.6f"),
                   "lon": st.column_config.NumberColumn("経度", format="%.6f"),
                   "自転車": st.column_config.NumberColumn("電動アシスト自転車（台）", min_value=0, step=1, format="%d"),
                   "キックボード": st.column_config.NumberColumn("電動キックボード（台）", min_value=0, step=1, format="%d"),
                   "収納": st.column_config.NumberColumn("収納可能台数", min_value=1, step=1, format="%d",
                                                        help="ポートに置ける最大台数（区画数）。満車なら返却失敗")})
edited = edited.dropna(subset=["lat", "lon"]).reset_index(drop=True)
for c in ("自転車", "キックボード"):
    edited[c] = edited[c].fillna(0).astype(int)
edited["収納"] = edited["収納"].fillna(DEFAULT_CAP).astype(int)
edited["name"] = [n if isinstance(n, str) and n else f"地点{i + 1}" for i, n in enumerate(edited["name"])]
over = edited["自転車"] + edited["キックボード"] > edited["収納"]
if over.any():
    st.warning("台数が収納可能台数を超えています（初期配置で返却失敗が起きやすくなります）："
               + "、".join(edited.loc[over, "name"].astype(str)))

s1, s2, t1, t2, t3 = st.columns([1, 1, 1, 1, 1.4])
st_eb = s1.number_input("駅ポート初期：自転車", 0, 200, int(ss.ev2_station[0]), key=f"ev2_seb_{ss.ev2_ver}")
st_kb = s2.number_input("駅ポート初期：キックボード", 0, 200, int(ss.ev2_station[1]), key=f"ev2_skb_{ss.ev2_ver}")
nb_tot = t1.number_input("自動配分：自転車合計", 0, 300, 15, key="ev2_nb")
nk_tot = t2.number_input("自動配分：キックボード合計", 0, 300, 10, key="ev2_nk")
with t3:
    st.write("")
    auto = st.button("台数を自動配分（駅ポート含む）", use_container_width=True, disabled=edited.empty)

# ---------- 計算 ----------
result = None
if not edited.empty:
    snapped = snap_ports(NET, edited)
    far = snapped["snap_m"] > MAX_SNAP_M
    if far.any():
        st.warning(f"道路網から{MAX_SNAP_M}m以上離れた地点は計算から除外しました：" + "、".join(snapped.loc[far, "name"]))
    use = snapped[~far].reset_index(drop=True)
    if len(use):
        L, assign = custom_layout(NET, cfg, use, sh, ops_spec.ebike_share, station_cap=ops_spec.station_cap)
        if auto:
            F0 = allocate_fleet(L, ops, int(nb_tot), int(nk_tot), oh, rh)
            set_ports(pd.concat([use.assign(自転車=F0.init[:L.k, EB], キックボード=F0.init[:L.k, KB]), snapped[far]])[COLS],
                      station=(int(F0.init[L.k, EB]), int(F0.init[L.k, KB])))
        init = np.zeros((L.k + 2, 2), dtype=int)
        init[:L.k, EB], init[:L.k, KB] = use["自転車"], use["キックボード"]
        init[L.k] = (st_eb, st_kb)
        F = Fleet(init)
        if sum(F.n) == 0:
            st.info("台数を入力してください。")
        else:
            _wp = weather_path(cfg)
            key = json.dumps([DATA_KEY, _wp.stat().st_mtime if _wp.exists() else 0, cfg, spec.key(), ops_spec.key(),
                              use[COLS].to_dict("list"), init.tolist()],
                             ensure_ascii=False, default=str)

            @st.cache_data(show_spinner="シミュレーション中…")
            def evaluate(key_):
                ms, dl = month_eval(L, F, cfg, DEM, month=ops_spec.month, seed=1, cal=CAL, **OPS_KW)
                return ms, dl, annual_cost(cfg, F, L.k, dl, ops_spec.station_cap)

            m_sum, daily, cost = evaluate(key)
            result = m_sum

# ---------- 結果 ----------
if result is not None:
    n_hol = int((CAL["曜日区分"] == "休日").sum())
    period_txt = f"{ops_spec.year}年{ops_spec.month}月の1か月（平日{len(CAL) - n_hol}日・休日{n_hol}日、{SOURCE}）"
    st.subheader(f"結果（{period_txt}）")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("充足率", f"{result['充足率']:.1%}", help=f"貸出失敗率 {result['貸出失敗率']:.1%}（往路・復路の合計）")
    c2.metric("合計移動時間", f"{result['合計移動時間(人・分/日)']:,.0f} 人・分/日",
              delta=f"{result['平均移動時間(分/片道)'] - result['全員徒歩の平均(分/片道)']:+.1f} 分/片道（全員徒歩比）",
              delta_color="inverse")
    c3.metric("返却失敗率", f"{result['返却失敗率']:.1%}", help="到着ポートが満車で返却できなかった割合（区画外に一時駐車）")
    c4.metric("年間総費用", f"{cost['年間総費用'] / 1e4:,.0f} 万円", help=f"1トリップあたり {cost['1トリップあたり費用']:,.0f} 円")
    nb, nk = F.n
    st.caption(f"台数：自転車{nb}＋キックボード{nk}＝{nb + nk}台（うち駅ポート初期 {st_eb + st_kb}台）・"
               f"代替乗車率 {result['代替乗車率']:.1%}・駅ポート最大在庫 {result['駅ポート最大在庫']}台／容量{ops_spec.station_cap}台・"
               f"リバランス {result['作業回数/日']:.1f}回/日（{result['移動台数/日']:.1f}台・{result['作業時間(分/日)']:.0f}分）・"
               f"電池交換 {result['電池交換/日']:.2f}回/日・稼働率 {result['稼働率(回/台/日)']:.2f}回/台/日")

# ---------- 地図（クリックでポート追加） ----------
st.subheader("地図（クリックで住宅側ポートを追加）")
n1, n2, n3, _ = st.columns([1, 1, 1, 1])
new_b = n1.number_input("追加するポート：自転車", 0, 100, 3, key="ev2_newb")
new_k = n2.number_input("追加するポート：キックボード", 0, 100, 2, key="ev2_newk")
new_cap = n3.number_input("追加するポート：収納可能台数", 1, 200, DEFAULT_CAP, key="ev2_newcap",
                          help="ポートに置ける最大台数（区画数）。満車のポートには返却できません。")
if result is not None:
    main = use.rename(columns={"name": "label"}).assign(
        初期台数=use["自転車"] + use["キックボード"], 必要区画=use["収納"],
        内訳=[f"自{b}・キ{k}" for b, k in zip(use["自転車"], use["キックボード"])])
    asg = assign
else:
    main = pd.DataFrame(columns=["label", "lat", "lon", "初期台数", "必要区画", "収納"])
    asg = np.zeros(len(NET.houses), dtype=int)
dest_tbl = pd.DataFrame([("駅", "station", int(st_eb + st_kb), 0, ops_spec.station_cap),
                         ("モビリティステーション", "aeon", 0, 0, None)],
                        columns=["ポート", "種別", "初期台数", "必要区画", "収納"])
st.caption("ラベル：自＝電動アシスト自転車、キ＝電動キックボードの初期台数 / 収納可能台数")
m = viz.port_map(NET.houses, NET.candidates, main, asg, dest_tbl, cfg, main_name="選択したポート", square=True,
                 color_houses=result is not None)
out = st_folium(m, height=560, use_container_width=True, key="ev2_map", returned_objects=["last_clicked"])
click = (out or {}).get("last_clicked")
if click and (click["lat"], click["lng"]) != ss.ev2_last_click:
    ss.ev2_last_click = (click["lat"], click["lng"])
    pt_df = pd.DataFrame({"name": [next_name(edited)], "lat": [click["lat"]], "lon": [click["lng"]]})
    sn = snap_ports(NET, pt_df).iloc[0]
    if sn["snap_m"] > MAX_SNAP_M:
        st.warning("クリック位置が街区の道路網から離れすぎています。")
    else:
        node = NET.nodes.loc[sn["node"]]
        set_ports(pd.concat([edited, pd.DataFrame({"name": [sn["name"]], "lat": [node["lat"]], "lon": [node["lon"]],
                                                   "自転車": [int(new_b)], "キックボード": [int(new_k)],
                                                   "収納": [int(new_cap)]})], ignore_index=True),
                  station=(int(st_eb), int(st_kb)))

# ---------- 詳細 ----------
if result is not None:
    a, b = st.columns([2, 3])
    with a:
        st.markdown("**ポートごと**")
        peaks = result["ポート別最大駐車(95%点)"]
        bp = pd.DataFrame({"ポート": list(use["name"]) + ["駅西口ポート"],
                           "自転車": list(use["自転車"]) + [st_eb], "キックボード": list(use["キックボード"]) + [st_kb],
                           "収納": list(use["収納"]) + [ops_spec.station_cap],
                           "最大駐車(95%点)": peaks[:L.k + 1],
                           "カバー戸数": list(np.bincount(assign, minlength=L.k)) + [None]})
        st.dataframe(bp, hide_index=True, use_container_width=True)
        st.caption("最大駐車が収納と同じなら、満車による返却失敗が起きています。")
        if st.button("この配置と結果を data/output に保存"):
            name = f"evaluated_mix_{ops_spec.pattern}_conv{round(conv * 100)}"
            csv, gj = save_ports(bp.iloc[:L.k].assign(緯度=use["lat"], 経度=use["lon"]).rename(columns={"ポート": "name"}),
                                 name, cfg)
            st.success(f"保存しました：`{csv.relative_to(ROOT)}` と `{gj.relative_to(ROOT)}`")
    with b:
        st.markdown("**時間帯別の利用希望と充足**")
        hr = np.array(result["_hourly"])
        if hr.size:
            st.plotly_chart(viz.hourly_service(pd.DataFrame({"時": np.arange(24), "利用希望": hr[:, 0], "利用": hr[:, 1]})),
                            use_container_width=True)
        keys = ["車両（償却）", "車両維持", "ポート（償却）", "用地", "人件費（リバランス）", "軽トラック"]
        st.markdown("**年間費用の内訳**")
        st.dataframe(pd.DataFrame({"項目": keys + ["年間総費用"], "円/年": [cost[x] for x in keys + ["年間総費用"]]})
                     .style.format({"円/年": "{:,.0f}"}), hide_index=True, use_container_width=True)
else:
    st.info("地図をクリックするか、「最適配置を読み込む」「ファイルから読み込む」でポートを置いてください。")
