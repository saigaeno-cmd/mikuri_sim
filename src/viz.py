"""Plotly グラフと folium 地図。"""
from __future__ import annotations

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .geo import haversine

# 既定の検証済みカテゴリ配色（固定順。順位ではなく対象に色を結びつける）
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY = "#8a8984"
DEST_COLOR = {"駅": SERIES[0], "イオン": SERIES[1]}
PART_COLOR = {"徒歩（自宅→ポート）": SERIES[0], "走行": SERIES[1], "徒歩（ポート→施設）": SERIES[2]}


def _style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=70, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0, title=None),
                      bargap=0.25)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False, rangemode="tozero")
    fig.update_traces(selector=dict(type="scatter"), line_width=2, marker_size=8)
    fig.update_traces(selector=dict(type="bar"), marker_line_width=0)
    return fig


def hourly_demand(demand: pd.DataFrame) -> go.Figure:
    d = demand.assign(
        目的地=demand["dest"].map({"station": "駅", "aeon": "イオン"}),
        方向=demand["direction"].map({"outbound": "往路（自宅→施設）", "return": "復路（施設→自宅）"}),
    ).groupby(["hour", "目的地", "方向"], as_index=False)["trips"].sum()
    fig = px.bar(d, x="hour", y="trips", color="目的地", facet_row="方向", color_discrete_map=DEST_COLOR,
                 labels={"hour": "時間帯（時）", "trips": "トリップ/時"}, category_orders={"目的地": ["駅", "イオン"]})
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(dtick=2, range=[-0.5, 23.5])
    # 分割グラフの自動レンジが積み上げ合計を拾わないことがあるため、明示する
    ymax = float(d.groupby(["hour", "方向"])["trips"].sum().max()) if len(d) else 1.0
    fig = _style(fig, 420)
    fig.update_yaxes(range=[0, ymax * 1.1], autorange=False)
    return fig


def fleet_by_ports(df: pd.DataFrame) -> go.Figure:
    """横軸ポート数、線＝転換率、列＝車種。"""
    d = df.assign(転換率=(df["転換率"] * 100).round().astype(int).astype(str) + "%")
    order = sorted(d["転換率"].unique(), key=lambda s: int(s[:-1]))
    fig = px.line(d, x="ポート数", y="必要台数", color="転換率", facet_col="車種", markers=True,
                  color_discrete_sequence=SERIES, category_orders={"転換率": order})
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(dtick=1)
    # 各線の右端に直接ラベル
    for tr in fig.data:
        tr.update(mode="lines+markers+text", text=[""] * (len(tr.x) - 1) + [tr.name],
                  textposition="middle right", textfont=dict(size=12))
    fig = _style(fig, 420)
    fig.update_layout(margin_t=100, legend_y=1.16, legend_title_text="利用転換率")
    return fig


def fleet_by_conversion(df: pd.DataFrame, k: int) -> go.Figure:
    d = df[df["ポート数"] == k].assign(転換率=lambda x: (x["転換率"] * 100).round().astype(int))
    fig = px.line(d, x="転換率", y="必要台数", color="車種", markers=True, color_discrete_sequence=SERIES,
                  labels={"転換率": "利用転換率（%）"})
    fig.update_xaxes(tickvals=sorted(d["転換率"].unique()), ticksuffix="%")
    return _style(fig, 340)


def fleet_breakdown(df: pd.DataFrame, vehicle_label: str, conversion: float) -> go.Figure:
    """場所ごとの必要区画（駐車スペース）。街区側は全ポートの合計。"""
    d = df[(df["車種"] == vehicle_label) & np.isclose(df["転換率"], conversion)].rename(
        columns={"街区側区画": "街区側（合計）", "駅ポート区画": "駅", "イオン区画": "イオン"})
    long = d.melt(id_vars="ポート数", value_vars=["街区側（合計）", "駅", "イオン"], var_name="場所", value_name="区画")
    fig = px.bar(long, x="ポート数", y="区画", color="場所", barmode="group", color_discrete_sequence=SERIES,
                 labels={"区画": "必要区画（台分）"})
    fig.update_xaxes(dtick=1)
    return _style(fig, 340)


def time_by_ports(loc: pd.DataFrame, share: float) -> go.Figure:
    d = loc[np.isclose(loc["駅の割合"], share)]
    fig = px.line(d, x="ポート数", y="合計(分)", color="車種", markers=True, color_discrete_sequence=SERIES,
                  labels={"合計(分)": "平均所要時間（分/トリップ）"})
    fig.update_xaxes(dtick=1)
    return _style(fig, 340)


def time_parts(loc: pd.DataFrame, vehicle: str, share: float) -> go.Figure:
    d = loc[(loc["vehicle"] == vehicle) & np.isclose(loc["駅の割合"], share)].rename(columns={
        "徒歩(分)": "徒歩（自宅→ポート）", "走行(分)": "走行", "目的地側徒歩(分)": "徒歩（ポート→施設）"})
    long = d.melt(id_vars="ポート数", value_vars=list(PART_COLOR), var_name="内訳", value_name="分")
    fig = px.bar(long, x="ポート数", y="分", color="内訳", color_discrete_map=PART_COLOR,
                 labels={"分": "平均時間（分/トリップ）"})
    fig.update_xaxes(dtick=1)
    fig.update_traces(marker_line_width=2, marker_line_color="rgba(255,255,255,0.9)")
    return _style(fig, 340)


def port_shift(loc: pd.DataFrame, k: int, candidates: pd.DataFrame, ref_ports: dict[str, tuple]) -> pd.DataFrame:
    """駅の割合を変えたとき、最適ポートが基準解からどれだけ動くか（各ポートの最近傍距離の平均, m）。"""
    lat, lon = candidates["lat"].to_numpy(), candidates["lon"].to_numpy()
    rows = []
    for r in loc[loc["ポート数"] == k].to_dict("records"):
        ref = np.array(ref_ports[r["vehicle"]])
        cur = np.array(r["ports"])
        D = haversine(lat[cur][:, None], lon[cur][:, None], lat[ref][None, :], lon[ref][None, :])
        rows.append({"車種": r["車種"], "駅の割合": r["駅の割合"], "合計(分)": r["合計(分)"],
                     "ずれ(m)": float(D.min(1).mean())})
    return pd.DataFrame(rows)


def share_sensitivity(shift: pd.DataFrame, y: str, ylab: str) -> go.Figure:
    d = shift.assign(駅の割合=(shift["駅の割合"] * 100).round().astype(int))
    fig = px.line(d, x="駅の割合", y=y, color="車種", markers=True, color_discrete_sequence=SERIES,
                  labels={"駅の割合": "目的地のうち駅の割合（%）", y: ylab})
    fig.update_xaxes(ticksuffix="%", dtick=10)
    return _style(fig, 320)


def port_layer_df(candidates: pd.DataFrame, sol_ports, fleet_table: pd.DataFrame) -> pd.DataFrame:
    """最適解（候補 index）→ 地図用の表（label, lat, lon, 初期台数, 必要区画）。"""
    nb = fleet_table[fleet_table["種別"] == "街区"].reset_index(drop=True)
    c = candidates.loc[list(sol_ports), ["lat", "lon"]].reset_index(drop=True)
    return c.assign(label=[f"P{j + 1}" for j in range(len(c))], 初期台数=nb["初期台数"], 必要区画=nb["必要区画"])


def _badge(text: str, col: str, sub: str, square: bool = False) -> str:
    """ポートの名札（上：名称、下：台数 / 収納）。近接したポートでも重なりにくい縦2段。"""
    radius = "2px" if square else "6px"
    return (f'<div style="display:flex;flex-direction:column;align-items:center;font:600 11px/1.15 sans-serif;'
            f'color:#0b0b0b;transform:translate(-50%,-50%);width:max-content">'
            f'<span style="background:{col};color:#fff;border:2px solid #fff;border-radius:{radius};'
            f'padding:0 4px;box-shadow:0 0 0 1px {col};white-space:nowrap">{text}</span>'
            f'<span style="background:rgba(255,255,255,.9);padding:0 3px;border-radius:3px;margin-top:1px;'
            f'white-space:nowrap;font-weight:600">{sub}</span></div>')


def port_map(houses: pd.DataFrame, candidates: pd.DataFrame, main: pd.DataFrame, assign: np.ndarray,
             fleet_table: pd.DataFrame, cfg: dict, *, main_name: str = "最適ポート", square: bool = False,
             overlays: dict[str, pd.DataFrame] | None = None, color_houses: bool = True,
             zoom: int = 17) -> folium.Map:
    """main: 住宅の色分けに使う配置（label, lat, lon, 初期台数, 必要区画）。
    overlays: {レイヤ名: DataFrame(label, lat, lon)} 比較用に破線の輪で重ねる配置。"""
    pts_all = np.vstack([houses[["lat", "lon"]].to_numpy(),
                         [[d["port"]["lat"], d["port"]["lon"]] for d in cfg["destinations"].values()]])
    center = [(pts_all[:, 0].min() + pts_all[:, 0].max()) / 2, (pts_all[:, 1].min() + pts_all[:, 1].max()) / 2]
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="背景地図（OSM）").add_to(m)

    fg_c = folium.FeatureGroup(name="ポート候補地点", show=False)
    for r in candidates.itertuples():
        folium.CircleMarker([r.lat, r.lon], radius=2, color=GRAY, fill=True, fill_opacity=0.6, weight=0,
                            tooltip=f"候補 {r.cand_id}").add_to(fg_c)
    fg_c.add_to(m)

    labels = list(main["label"])
    fg_h = folium.FeatureGroup(name=f"住宅（{main_name}への割当）")
    for r, a in zip(houses.itertuples(), assign):
        colored = color_houses and len(labels) > 0
        folium.CircleMarker([r.lat, r.lon], radius=4, color="white", weight=1, fill=True,
                            fill_color=SERIES[a % len(SERIES)] if colored else GRAY, fill_opacity=0.9,
                            tooltip=f"住宅 {r.house_id}" + (f" → {labels[a]}" if colored else "")).add_to(fg_h)
    fg_h.add_to(m)

    fg_p = folium.FeatureGroup(name=main_name)
    for j, r in enumerate(main.to_dict("records")):
        n, slots = int(r["初期台数"]), int(r["必要区画"])
        cap = r.get("収納")
        has_cap = cap is not None and not (isinstance(cap, float) and np.isnan(cap))
        sub = f"{n}台/収納{int(cap)}" if has_cap else f"{n}台"
        if r.get("内訳"):
            sub = f"{r['内訳']}" + (f"/収納{int(cap)}" if has_cap else "")
        tip = f"{r['label']}: 初期 {n}台" + (f" / 収納可能 {int(cap)}台" if has_cap else f" / 区画 {slots}")
        folium.Marker([r["lat"], r["lon"]], tooltip=tip,
                      icon=folium.DivIcon(icon_size=(0, 0), icon_anchor=(0, 0),
                                          html=_badge(r["label"], SERIES[j % len(SERIES)], sub, square))
                      ).add_to(fg_p)
    fg_p.add_to(m)

    for name, df in (overlays or {}).items():
        fg = folium.FeatureGroup(name=f"比較：{name}")
        for r in df.to_dict("records"):
            folium.CircleMarker([r["lat"], r["lon"]], radius=11, color="#0b0b0b", weight=2, dash_array="4 3",
                                fill=False, tooltip=f"{name}：{r['label']}").add_to(fg)
        fg.add_to(m)

    fg_d = folium.FeatureGroup(name="目的地ポート")
    dest_row = {k: d.get("port_label", k) for k, d in cfg["destinations"].items()}
    for key, d in cfg["destinations"].items():
        row = fleet_table[fleet_table["種別"] == key]
        info = f"{d['label']}：初期 {int(row['初期台数'].iloc[0])}台 / 必要区画 {int(row['必要区画'].iloc[0])}" if len(row) else d["label"]
        folium.Marker([d["port"]["lat"], d["port"]["lon"]], tooltip=info,
                      icon=folium.Icon(color="darkblue" if key == "station" else "orange",
                                       icon="train" if key == "station" else "shopping-cart", prefix="fa")).add_to(fg_d)
        cap_txt = ""
        if "収納" in fleet_table.columns and len(row):
            cv = row["収納"].iloc[0]
            ini = int(row["初期台数"].iloc[0]) if "初期台数" in row else 0
            head = f"初期{ini}台・" if ini > 0 else ""
            cap_txt = f"（{head}収納{int(cv)}）" if cv is not None and not pd.isna(cv) else f"（{head}収納 制限なし）"
        folium.Marker([d["port"]["lat"], d["port"]["lon"]], icon=folium.DivIcon(
            icon_size=(200, 20), icon_anchor=(-14, 30),
            html=f'<div style="font:600 12px sans-serif;background:rgba(255,255,255,.85);padding:0 4px;'
                 f'border-radius:3px;width:max-content">{dest_row[key]}{cap_txt}</div>')).add_to(fg_d)
    fg_d.add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
    return m


def station_slots_chart(df: pd.DataFrame, cap: int | None) -> go.Figure:
    d = df[(df["ポート数"] == df["ポート数"].min()) & (df["車種"] == df["車種"].iloc[0])]
    d = d.assign(転換率=(d["転換率"] * 100).round().astype(int))
    over = d["駅ポート区画"] > (cap or np.inf)
    fig = go.Figure(go.Bar(
        x=d["転換率"], y=d["駅ポート区画"], marker_color=np.where(over, SERIES[7], SERIES[0]),
        text=[f"{v}台分" + (" 超過" if o else "") for v, o in zip(d["駅ポート区画"], over)],
        textposition="outside", width=6, hovertemplate="転換率 %{x}%<br>必要区画 %{y}台分<extra></extra>"))
    if cap:
        fig.add_hline(y=cap, line_dash="dash", line_color=GRAY, line_width=2,
                      annotation_text=f"上限 {cap}台（幅{cap}m）", annotation_position="top left")
    fig.update_xaxes(tickvals=d["転換率"], ticksuffix="%", title="利用転換率")
    fig.update_yaxes(title="駅ポート必要区画（台分）")
    fig = _style(fig, 320)
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def fulfill_vs_fleet(sweep: pd.DataFrame, target: float, best: tuple[int, int] | None = None) -> go.Figure:
    """横軸 総台数、縦軸 充足率、線＝ポート数。"""
    d = sweep.assign(ポート数=sweep["ポート数"].astype(str) + "か所", 充足率=sweep["充足率"] * 100)
    order = [f"{k}か所" for k in sorted(sweep["ポート数"].unique())]
    fig = px.line(d, x="台数", y="充足率", color="ポート数", markers=True, color_discrete_sequence=SERIES,
                  category_orders={"ポート数": order}, labels={"台数": "総台数（街区側）", "充足率": "充足率（%）"})
    fig.add_hline(y=target * 100, line_dash="dash", line_color=GRAY, line_width=2,
                  annotation_text=f"目標 {target:.0%}", annotation_position="top left")
    if best:
        b = sweep[(sweep["ポート数"] == best[0]) & (sweep["台数"] == best[1])]
        if len(b):
            fig.add_scatter(x=b["台数"], y=b["充足率"] * 100, mode="markers", showlegend=False,
                            marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color="#0b0b0b", width=2)),
                            hovertemplate="推奨<extra></extra>")
    fig = _style(fig, 360)
    fig.update_yaxes(range=[0, 105], ticksuffix="%")
    fig.update_layout(legend_title_text="ポート数")
    return fig


def time_vs_fleet(sweep: pd.DataFrame) -> go.Figure:
    d = sweep.assign(ポート数=sweep["ポート数"].astype(str) + "か所")
    order = [f"{k}か所" for k in sorted(sweep["ポート数"].unique())]
    fig = px.line(d, x="台数", y="平均移動時間", color="ポート数", markers=True, color_discrete_sequence=SERIES,
                  category_orders={"ポート数": order},
                  labels={"台数": "総台数（街区側）", "平均移動時間": "平均移動時間（分/片道）"})
    fig = _style(fig, 360)
    fig.update_layout(legend_title_text="ポート数")
    return fig


def hourly_service(hourly: pd.DataFrame) -> go.Figure:
    """時間帯別の利用（往路・復路）と断念の積み上げ。"""
    h = hourly.copy()
    if "往路希望" not in h:  # 旧形式
        h["往路希望"], h["往路利用"], h["復路希望"], h["復路利用"] = h["利用希望"], h["利用"], 0.0, 0.0
    h["希望"] = h["往路希望"] + h["復路希望"]
    h = h[h["希望"] > 0.001]
    fig = go.Figure()
    fig.add_bar(x=h["時"], y=h["往路利用"], name="利用できた（往路）", marker_color=SERIES[0],
                hovertemplate="%{x}時台 往路 利用 %{y:.1f}件<extra></extra>")
    fig.add_bar(x=h["時"], y=h["復路利用"], name="利用できた（復路）", marker_color=SERIES[2],
                hovertemplate="%{x}時台 復路 利用 %{y:.1f}件<extra></extra>")
    fig.add_bar(x=h["時"], y=h["希望"] - h["往路利用"] - h["復路利用"], name="利用できなかった", marker_color=GRAY,
                hovertemplate="%{x}時台 断念 %{y:.1f}件<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_xaxes(dtick=1, title="時間帯（時）", range=[4.5, 23.5])
    fig.update_yaxes(title="利用希望（件/日）",
                     range=[0, float(h["希望"].max()) * 1.15 if len(h) else 1], autorange=False)
    fig = _style(fig, 320)
    fig.update_traces(marker_line_width=0)
    return fig
