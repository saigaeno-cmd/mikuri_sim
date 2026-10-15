"""添付されたポート設置場所の読込と評価（事前計算済みの全ノード行列を使う。アプリで道路網計算はしない）。"""
from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from .demand import DESTS, build_demand, dest_shares, hourly_matrix
from .fleet import simulate_fleet
from .geo import haversine
from .loaders import Network
from .location import Solution, cost_matrix
from .sensitivity import Scenario

LAT = ["lat", "latitude", "緯度", "y"]
LON = ["lon", "lng", "long", "longitude", "経度", "x"]
NAME = ["name", "名称", "ポート名", "port", "label", "id", "番号"]
ACCEPT = ["csv", "geojson", "json", "gpkg", "gpx", "kml"]


def _pick(cols, cands):
    low = {str(c).strip().lower(): c for c in cols}
    return next((low[c.lower()] for c in cands if c.lower() in low), None)


def _from_csv(data: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp932"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("CSVの文字コードを判別できません（UTF-8 / Shift_JIS に対応）")
    la, lo = _pick(df.columns, LAT), _pick(df.columns, LON)
    if la is None or lo is None:
        raise ValueError(f"CSVに緯度・経度の列が見つかりません（lat/lon または 緯度/経度）。列: {list(df.columns)}")
    nm = _pick(df.columns, NAME)
    return pd.DataFrame({"name": df[nm].astype(str) if nm else None,
                         "lat": pd.to_numeric(df[la], errors="coerce"),
                         "lon": pd.to_numeric(df[lo], errors="coerce")})


def _from_gpx(data: bytes) -> pd.DataFrame:
    """ウェイポイント（wpt）。無ければ各トラックの始点。"""
    root = ET.fromstring(data)
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    rows = [(w.findtext(f"{ns}name"), float(w.get("lat")), float(w.get("lon"))) for w in root.iter(f"{ns}wpt")]
    if not rows:
        for trk in root.iter(f"{ns}trk"):
            p = next(trk.iter(f"{ns}trkpt"), None)
            if p is not None:
                rows.append((trk.findtext(f"{ns}name"), float(p.get("lat")), float(p.get("lon"))))
    return pd.DataFrame(rows, columns=["name", "lat", "lon"])


def _from_geo(data: bytes, suffix: str) -> pd.DataFrame:
    import geopandas as gpd

    g = gpd.read_file(io.BytesIO(data)) if suffix != "kml" else gpd.read_file(io.BytesIO(data), driver="KML")
    if g.crs is not None and g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    pts = g.geometry.where(g.geometry.geom_type == "Point", g.geometry.representative_point())
    nm = _pick([c for c in g.columns if c != "geometry"], NAME)
    return pd.DataFrame({"name": g[nm].astype(str).to_numpy() if nm else None,
                         "lat": pts.y.to_numpy(), "lon": pts.x.to_numpy()})


def parse_ports(filename: str, data: bytes) -> pd.DataFrame:
    """添付ファイル → name, lat, lon。"""
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix == "csv":
        df = _from_csv(data)
    elif suffix == "gpx":
        df = _from_gpx(data)
    elif suffix in ("geojson", "json", "gpkg", "kml"):
        df = _from_geo(data, suffix)
    else:
        raise ValueError(f"対応していない形式です: .{suffix}（{' / '.join(ACCEPT)}）")
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("ファイルからポート位置（点）を読み取れませんでした")
    if df["lat"].abs().max() > 90 and df["lon"].abs().max() <= 90:  # 緯度経度の取り違え
        df[["lat", "lon"]] = df[["lon", "lat"]].to_numpy()
    df["name"] = [n if isinstance(n, str) and n not in ("", "None", "nan") else f"指定{i + 1}"
                  for i, n in enumerate(df["name"])]
    return df


def snap_ports(net: Network, ports: pd.DataFrame) -> pd.DataFrame:
    """最寄りの道路ノードに合わせる。node=ノード番号、snap_m=ノードまでの直線距離。"""
    nodes = net.nodes[net.nodes["usable"]]
    D = haversine(ports["lat"].to_numpy()[:, None], ports["lon"].to_numpy()[:, None],
                  nodes["lat"].to_numpy()[None, :], nodes["lon"].to_numpy()[None, :])
    j = D.argmin(1)
    return ports.assign(node=nodes.index.to_numpy()[j], snap_m=D[np.arange(len(ports)), j].round(1))


def evaluate_ports(net: Network, survey, cfg: dict, ports: pd.DataFrame, *, conversion, purposes,
                   station_share, vehicle, opts=None) -> Scenario:
    """指定したポート配置（snap_ports 済み）での割当・所要時間・必要台数。"""
    idx = [net.dest_keys.index(d) for d in DESTS]
    node, off = ports["node"].to_numpy(), ports["snap_m"].to_numpy()
    W = net.Wn[:, node] + off[None, :]
    R = net.Rn[node][:, idx] + off[:, None]
    demand = build_demand(survey, cfg, conversion, purposes, station_share, **(opts or {}))
    shares = dest_shares(demand)
    C, walk, ride, egress = cost_matrix(W, R, net.E[idx], shares, cfg["walk_speed_kmh"],
                                        cfg["vehicles"][vehicle]["speed_kmh"])
    assign = C.argmin(1)
    rows = np.arange(len(assign))
    sol = Solution(ports=np.arange(len(ports)), assign=assign, objective=float(C[rows, assign].sum()))
    port_share = np.bincount(assign, minlength=len(ports)) / len(assign)
    fleet = simulate_fleet(hourly_matrix(demand, "outbound"), hourly_matrix(demand, "return"),
                           port_share, cfg["simulation"]["service_level"], port_labels=list(ports["name"]))
    return Scenario(demand, shares, sol, fleet, float(walk[rows, assign].mean()),
                    float(ride[assign].mean()), egress)


def custom_walk_m(net: Network, ports: pd.DataFrame, assign: np.ndarray) -> np.ndarray:
    """住宅ごとの、割当てた添付ポートまでの徒歩距離（m）。"""
    node, off = ports["node"].to_numpy(), ports["snap_m"].to_numpy()
    return net.Wn[np.arange(len(assign)), node[assign]] + off[assign]
