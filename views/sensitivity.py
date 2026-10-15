"""感度分析ページ：転換率・ポート数ごとの必要台数とポート位置。"""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src import viz
from src.custom_ports import ACCEPT, custom_walk_m, evaluate_ports, parse_ports, snap_ports
from src.export import ports_table, save_ports
from src.loaders import ROOT
from src.ui import demand_sidebar, header_warnings, load_or_stop, vehicle_radio
from src.sensitivity import max_conversion_within, run_scenario, sweep_fleet, sweep_location

BASE_CFG, SURVEY, NET, FLAGS, DATA_KEY = load_or_stop()


@st.cache_data(show_spinner="必要台数の感度分析を計算中…")
def cached_fleet_sweep(cfg_json: str, purposes: tuple, share: float | None, opts_json: str, _key: str):
    cfg = json.loads(cfg_json)
    return sweep_fleet(NET, SURVEY, cfg, purposes=list(purposes), station_share=share,
                       vehicles=list(cfg["vehicles"]), ks=range(1, cfg["simulation"]["max_ports"] + 1),
                       conversions=cfg["simulation"]["conversion_rates"], opts=_opts(opts_json))


@st.cache_data(show_spinner="ポート位置の感度分析を計算中…")
def cached_location_sweep(cfg_json: str, purposes: tuple, shares: tuple, opts_json: str, _key: str):
    cfg = json.loads(cfg_json)
    return sweep_location(NET, SURVEY, cfg, purposes=list(purposes), vehicles=list(cfg["vehicles"]),
                          ks=range(1, cfg["simulation"]["max_ports"] + 1), station_shares=list(shares),
                          opts=_opts(opts_json))


def _opts(opts_json: str) -> dict:
    o = json.loads(opts_json)
    if o.get("hourly") is not None:
        o["hourly"] = tuple(tuple(r) for r in o["hourly"])
    return o


# ---------- サイドバー ----------
cfg = copy.deepcopy(BASE_CFG)
sim = cfg["simulation"]
veh_keys = list(cfg["vehicles"])
with st.sidebar:
    st.header("パラメータ")
    vehicle = vehicle_radio(cfg, key="sens_vehicle")
    spec = demand_sidebar(cfg, SURVEY, key_prefix="sens_")
    conversion, purposes, station_share, opts = spec.conversion, spec.purposes, spec.station_share, spec.opts
    conv_pct, day, survey_share = round(conversion * 100), opts["day"], spec.survey_share

    k = st.slider("街区側ポート数", 1, sim["max_ports"], 3)
    cfg["station_port"]["max_slots"] = st.number_input(
        "駅ポートの容量（台）", 1, 200, int(cfg["station_port"]["max_slots"]), key="sens_cap",
        help="基本25台（区画①18＋区画②1列7）。区画①のみ18台、区画②も2列なら32台")

    with st.expander("詳細設定"):
        sim["service_level"] = st.slider("サービス水準（欠車しない確率）", 0.50, 0.99, sim["service_level"], 0.01)
        cfg["walk_speed_kmh"] = st.number_input("徒歩速度 km/h", 2.0, 8.0, cfg["walk_speed_kmh"], 0.1)
        for v in veh_keys:
            cfg["vehicles"][v]["speed_kmh"] = st.number_input(
                f"{cfg['vehicles'][v]['label']} km/h", 5.0, 30.0, float(cfg["vehicles"][v]["speed_kmh"]), 0.5)

cfg_json = json.dumps(cfg, ensure_ascii=False, sort_keys=True, default=str)
opts_json = json.dumps(opts, ensure_ascii=False)
eff_share = survey_share if station_share is None else station_share
grid = tuple(sorted(set(np.round(np.linspace(0, 1, 11), 2)) | {round(eff_share, 4)}))

# ---------- ヘッダ ----------
st.title("感度分析")
header_warnings(NET, FLAGS)

scn = run_scenario(NET, SURVEY, cfg, conversion=conversion, purposes=purposes,
                   station_share=station_share, vehicle=vehicle, k=k, opts=opts)
fleet_df = cached_fleet_sweep(cfg_json, tuple(purposes), station_share, opts_json, DATA_KEY)
loc_df = cached_location_sweep(cfg_json, tuple(purposes), grid, opts_json, DATA_KEY)

sp = cfg["station_port"]
width_m = scn.fleet.station_slots * sp["slot_width_m"]
area_m2 = width_m * sp["slot_depth_m"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("必要台数（合計）", f"{scn.fleet.total} 台")
c2.metric("平均所要時間", f"{scn.total_min:.1f} 分/トリップ",
          help="徒歩（自宅→ポート）＋走行＋徒歩（ポート→施設）")
cap = sp.get("max_slots")
over_n = scn.fleet.station_slots - cap if cap else 0
c3.metric("駅ポート必要区画", f"{scn.fleet.station_slots} 台分",
          delta=(f"上限{cap}台を {over_n} 超過" if over_n > 0 else f"上限{cap}台以内") if cap else None,
          delta_color="inverse" if over_n > 0 else "off",
          help="駅ポートに同時に駐車される最大台数（余裕込み）")
c4.metric("駅ポート必要幅（1列）", f"{width_m:.0f} m",
          help=f"1台 幅{sp['slot_width_m']} m × 奥行{sp['slot_depth_m']} m。1列に並べた場合。面積 {area_m2:.0f} m²")
if cap:
    c_max = max_conversion_within(SURVEY, cfg, cap, purposes=purposes, station_share=station_share, opts=opts)
    if scn.fleet.station_slots > cap:
        st.error(f"駅ポートの必要区画 {scn.fleet.station_slots} 台分が上限 {cap} 台（幅 {cap * sp['slot_width_m']:.0f} m）を"
                 f"超えています。この条件で上限内に収まる転換率は **約 {c_max:.1%} まで** です。", icon="🚫")
    else:
        st.success(f"駅ポートは上限 {cap} 台以内に収まります（上限内に収まる転換率は約 {c_max:.1%} まで）。", icon="✅")
st.caption(f"1日のトリップ数：{scn.demand['trips'].sum():.1f}（往復計）／ "
           f"車種：{cfg['vehicles'][vehicle]['label']} ／ 転換率 {conv_pct}% ／ 街区側ポート {k}か所")

tab_sum, tab_fleet, tab_loc, tab_map, tab_tbl = st.tabs(
    ["概要", "必要台数の感度分析", "ポート位置の感度分析", "地図", "結果表"])

with tab_sum:
    st.subheader("時間帯別の需要")
    st.plotly_chart(viz.hourly_demand(scn.demand), use_container_width=True)
    st.subheader("所要時間の内訳（現在の設定）")
    st.dataframe(pd.DataFrame({
        "項目": ["徒歩（自宅→ポート）", "走行", "徒歩（ポート→施設）", "合計"],
        "平均（分/トリップ）": [scn.walk_min, scn.ride_min, scn.egress_min, scn.total_min],
    }).round(2), hide_index=True)

with tab_fleet:
    st.subheader("ポート数・転換率ごとの必要台数")
    st.plotly_chart(viz.fleet_by_ports(fleet_df), use_container_width=True)
    a, b = st.columns(2)
    with a:
        st.markdown(f"**転換率ごとの必要台数（街区側ポート {k}か所）**")
        st.plotly_chart(viz.fleet_by_conversion(fleet_df, k), use_container_width=True)
    with b:
        st.markdown(f"**場所ごとの必要区画（{cfg['vehicles'][vehicle]['label']}・転換率 {conv_pct}%）**")
        st.plotly_chart(viz.fleet_breakdown(fleet_df, cfg["vehicles"][vehicle]["label"], conversion),
                        use_container_width=True)
    st.caption("必要台数＝各街区ポートに朝の時点で置いておく台数の合計（再配置なしの前提）。街区ポートを増やすと各ポートで個別に在庫を持つため、合計台数は増える傾向があります。"
               "1時間単位の在庫計算のため、車種（速度）の違いは主に所要時間とポート位置に表れます。")
    st.markdown("**駅ポートの必要区画と上限**（駅ポートの出入りは街区側ポート数・車種に依存しません）")
    st.plotly_chart(viz.station_slots_chart(fleet_df, cap), use_container_width=True)
    with st.expander("表で見る"):
        st.dataframe(fleet_df, hide_index=True, use_container_width=True)

with tab_loc:
    st.subheader("ポート数と平均所要時間")
    a, b = st.columns(2)
    with a:
        st.markdown("**車種別の平均所要時間**")
        st.plotly_chart(viz.time_by_ports(loc_df, round(eff_share, 4)), use_container_width=True)
    with b:
        st.markdown(f"**内訳（{cfg['vehicles'][vehicle]['label']}）**")
        st.plotly_chart(viz.time_parts(loc_df, vehicle, round(eff_share, 4)), use_container_width=True)

    st.subheader(f"目的地比率に対する感度（街区側ポート {k}か所）")
    ref = {r["vehicle"]: r["ports"] for r in loc_df[(loc_df["ポート数"] == k)
                                                    & np.isclose(loc_df["駅の割合"], round(eff_share, 4))]
           .to_dict("records")}
    shift = viz.port_shift(loc_df[loc_df["駅の割合"].isin(np.round(np.linspace(0, 1, 11), 2))],
                           k, NET.candidates, ref)
    a, b = st.columns(2)
    with a:
        st.plotly_chart(viz.share_sensitivity(shift, "合計(分)", "平均所要時間（分/トリップ）"),
                        use_container_width=True)
    with b:
        st.plotly_chart(viz.share_sensitivity(shift, "ずれ(m)", "現在の最適位置からのずれ（m）"),
                        use_container_width=True)
    st.caption("ずれ＝各ポートから、現在の設定での最適ポートのうち最も近いものまでの距離の平均。"
               "位置は需要の総量に依存しないため、転換率を変えても変わりません。")

with tab_map:
    st.subheader("ポート設置場所の添付")
    up_col, tpl_col = st.columns([3, 1])
    upload = up_col.file_uploader(
        "設置予定のポート位置を添付（CSV / GeoJSON / GPKG / GPX / KML）", type=ACCEPT,
        help="CSVは 緯度・経度（lat, lon）列が必須、名称（name）列は任意。GPXはウェイポイント（無ければ各トラックの始点）。"
             "面・線は代表点を使います。")
    tpl = viz.port_layer_df(NET.candidates, scn.solution.ports, scn.fleet.table)
    tpl_col.download_button("CSVのひな形（現在の最適配置）",
                            tpl.rename(columns={"label": "name"})[["name", "lat", "lon"]].to_csv(index=False)
                            .encode("utf-8-sig"), file_name="ports_template.csv", mime="text/csv",
                            use_container_width=True)

    custom = None
    if upload is not None:
        try:
            parsed = parse_ports(upload.name, upload.getvalue())
        except Exception as e:  # noqa: BLE001  読込エラーはそのまま利用者に見せる
            st.error(f"{upload.name} を読み込めませんでした：{e}")
            parsed = None
        if parsed is not None:
            parsed.insert(0, "使用", True)
            edited_ports = st.data_editor(
                parsed, key=f"ports_{upload.name}_{upload.size}", hide_index=True, num_rows="dynamic",
                use_container_width=True,
                column_config={"使用": st.column_config.CheckboxColumn(width="small"),
                               "name": st.column_config.TextColumn("名称"),
                               "lat": st.column_config.NumberColumn("緯度", format="%.6f"),
                               "lon": st.column_config.NumberColumn("経度", format="%.6f")})
            use = edited_ports[edited_ports["使用"].fillna(False) & edited_ports["lat"].notna()
                               & edited_ports["lon"].notna()].drop(columns="使用").reset_index(drop=True)
            if use.empty:
                st.warning("使用するポートがありません。")
            else:
                use["name"] = [n if isinstance(n, str) and n else f"指定{i + 1}" for i, n in enumerate(use["name"])]
                snapped = snap_ports(NET, use)
                far = snapped["snap_m"] > 300
                if far.any():
                    st.warning("街区の道路網から300m以上離れているため除外しました："
                               + "、".join(snapped.loc[far, "name"]))
                custom = snapped[~far].reset_index(drop=True)
                near = custom["snap_m"] > 50
                if near.any():
                    st.info("道路から50m以上離れた地点は、最寄りの道路ノード＋直線距離で評価しています："
                            + "、".join(custom.loc[near, "name"]))
                if custom.empty:
                    custom = None

    other = [v for v in veh_keys if v != vehicle]
    overlays = {}
    if custom is not None:
        m_ports = len(custom)
        cscn = evaluate_ports(NET, SURVEY, cfg, custom, conversion=conversion, purposes=purposes,
                              station_share=station_share, vehicle=vehicle, opts=opts)
        same = run_scenario(NET, SURVEY, cfg, conversion=conversion, purposes=purposes,
                            station_share=station_share, vehicle=vehicle,
                            k=min(m_ports, len(NET.candidates)), opts=opts)

        def _row(name, sc, walk_m):
            return {"配置": name, "ポート数": len(sc.solution.ports), "必要台数": sc.fleet.total,
                    "平均所要時間(分)": round(sc.total_min, 2), "うち徒歩(分)": round(sc.walk_min, 2),
                    "平均徒歩距離(m)": round(float(walk_m.mean())), "最大徒歩距離(m)": round(float(walk_m.max())),
                    "駅ポート区画": sc.fleet.station_slots}

        def _opt_walk(sc):
            return NET.W[np.arange(len(sc.solution.assign)), sc.solution.ports[sc.solution.assign]]

        rows = [_row("添付した配置", cscn, custom_walk_m(NET, custom, cscn.solution.assign)),
                _row(f"最適配置（同数 {len(same.solution.ports)}か所）", same, _opt_walk(same))]
        if k != len(same.solution.ports):
            rows.append(_row(f"最適配置（サイドバー {k}か所）", scn, _opt_walk(scn)))
        cmp_df = pd.DataFrame(rows)
        st.markdown(f"**配置の比較**（{cfg['vehicles'][vehicle]['label']}・転換率 {conv_pct}%）")
        st.dataframe(cmp_df, hide_index=True, use_container_width=True)
        gap = cscn.total_min - same.total_min
        st.caption(f"添付した配置は、同じ数の最適配置より平均所要時間が {gap:+.2f} 分/トリップ"
                   f"（{gap / same.total_min:+.1%}）。")

        color_by = st.radio("住宅の色分け", ["添付した配置", "最適配置"], horizontal=True)
        custom_main = custom.rename(columns={"name": "label"}).assign(
            初期台数=cscn.fleet.table.loc[cscn.fleet.table["種別"] == "街区", "初期台数"].to_numpy(),
            必要区画=cscn.fleet.table.loc[cscn.fleet.table["種別"] == "街区", "必要区画"].to_numpy())
        opt_main = viz.port_layer_df(NET.candidates, scn.solution.ports, scn.fleet.table)
        if color_by == "添付した配置":
            main, assign, table, main_name, square = custom_main, cscn.solution.assign, cscn.fleet.table, "添付したポート", True
            overlays[f"最適ポート（{k}か所）"] = opt_main
        else:
            main, assign, table, main_name, square = opt_main, scn.solution.assign, scn.fleet.table, "最適ポート", False
            overlays["添付したポート"] = custom_main
    else:
        main = viz.port_layer_df(NET.candidates, scn.solution.ports, scn.fleet.table)
        assign, table, main_name, square = scn.solution.assign, scn.fleet.table, "最適ポート", False
        st.caption("ファイルを添付すると、その位置を地図に表示し、最適配置と比較します。")

    if other and st.checkbox(f"{cfg['vehicles'][other[0]]['label']} の最適ポートを重ねる", value=custom is None):
        oc = NET.candidates.loc[list(ref[other[0]])].reset_index(drop=True)
        overlays[f"{cfg['vehicles'][other[0]]['label']} の最適ポート"] = oc.assign(
            label=[f"P{j + 1}" for j in range(len(oc))])
    m = viz.port_map(NET.houses, NET.candidates, main, assign, table, cfg,
                     main_name=main_name, square=square, overlays=overlays)
    st_folium(m, height=620, use_container_width=True, returned_objects=[])

    if custom is not None:
        st.markdown("**添付したポートごとの結果**")
        ct = cscn.fleet.table[cscn.fleet.table["種別"] == "街区"].reset_index(drop=True)
        ct = ct.assign(緯度=custom["lat"], 経度=custom["lon"], 道路までの距離_m=custom["snap_m"],
                       カバー戸数=np.bincount(cscn.solution.assign, minlength=len(custom)))
        ct["出発トリップ/日"] = ct["出発トリップ/日"].round(1)
        st.dataframe(ct.drop(columns="種別"), hide_index=True, use_container_width=True)
        if (ct["カバー戸数"] == 0).any():
            st.warning("どの住宅からも選ばれないポートがあります："
                       + "、".join(ct.loc[ct["カバー戸数"] == 0, "ポート"]))
        if st.button("添付した配置の評価を data/output に保存"):
            name = f"attached_ports_{vehicle}_conv{conv_pct}" + (f"_{day}" if day else "")
            csv, gj = save_ports(ct.rename(columns={"道路までの距離_m": "道路までの距離(m)"}), name, cfg)
            st.success(f"保存しました：`{csv.relative_to(ROOT)}` と `{gj.relative_to(ROOT)}`")

with tab_tbl:
    st.subheader("ポート別の台数")
    t = ports_table(scn, NET, cfg)
    st.dataframe(t, hide_index=True, use_container_width=True)
    name = f"ports_{vehicle}_k{k}_conv{conv_pct}" + (f"_{day}" if day else "")
    a, b = st.columns([1, 2])
    a.download_button("CSVをダウンロード", t.to_csv(index=False).encode("utf-8-sig"),
                      file_name=f"{name}.csv", mime="text/csv")
    if b.button(f"data/output/{name}.csv（.geojson）に保存"):
        csv, gj = save_ports(t, name, cfg)
        st.success(f"保存しました：`{csv.relative_to(ROOT)}` と `{gj.relative_to(ROOT)}`")
    with st.expander("事前計算の情報"):
        st.json(NET.meta)
