"""道路網を取得し、距離行列を data/processed/ に保存する（アプリ実行前に1回だけ）。

使い方:
    python scripts/precompute_network.py                     # config の osm_xml（data/raw/map.osm）があればそれを使う
    python scripts/precompute_network.py --source overpass   # osmnx で Overpass API から取得
    python scripts/precompute_network.py --source synthetic  # 格子状の合成ネットワーク（テスト専用）

出力:
    matrices.npz        W(住宅→候補 徒歩), R(候補→目的地ポート 走行), E(目的地ポート→施設入口 徒歩)  単位m
    houses_snapped.csv  住宅座標と最寄りノード
    candidates.csv      ポート候補地点
    walk.graphml / bike.graphml  （OSM取得時）
    meta.json           取得元・日時・入力ハッシュ
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import ConvexHull, cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.geo import LocalXY, haversine  # noqa: E402
from src.loaders import file_hash, houses_path, load_config, load_houses, resolve  # noqa: E402


def fetch_osm(bbox, network_type: str, cache: Path) -> nx.MultiDiGraph:
    import osmnx as ox

    if cache.exists():
        print(f"  キャッシュを使用: {cache.name}")
        return ox.load_graphml(cache)
    west, south, east, north = bbox
    print(f"  OSMから取得中: network_type={network_type}")
    G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type=network_type, simplify=True)
    ox.save_graphml(G, cache)
    return G


def graphs_from_xml(path: Path, cfg: dict) -> tuple[nx.MultiDiGraph, nx.MultiDiGraph]:
    """OSM XML から徒歩用・走行用のグラフを作る（Overpass 不要）。

    徒歩：highway タグのある道すべて（双方向）。
    走行：ride_exclude_highway を除き、歩道(footway)は ride_allow_footway（横断歩道）のみ。一方通行は守る。
    """
    import osmnx as ox

    if "footway" not in ox.settings.useful_tags_way:
        ox.settings.useful_tags_way = list(ox.settings.useful_tags_way) + ["footway"]
    G = ox.graph_from_xml(path, bidirectional=False, simplify=False, retain_all=True)
    G.remove_edges_from([(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if not d.get("highway")])
    G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])

    net = cfg["network"]
    excl = set(net["ride_exclude_highway"])
    allow_fw = set(net["ride_allow_footway"])

    def rideable(d):
        hw = d.get("highway")
        if hw in excl:
            return False
        return hw != "footway" or d.get("footway") in allow_fw

    walk = ox.truncate.largest_component(G, strongly=False)
    bike = G.edge_subgraph([(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if rideable(d)]).copy()
    bike = ox.truncate.largest_component(bike, strongly=True)
    return nx.MultiDiGraph(walk), nx.MultiDiGraph(bike)


def synthetic_grid(bbox, spacing_m: float = 40.0) -> nx.MultiDiGraph:
    """Overpass に接続できない環境で動作確認するための格子状道路網。実解析には使わないこと。"""
    west, south, east, north = bbox
    proj = LocalXY((south + north) / 2, (west + east) / 2)
    x0, y0 = proj.to_xy(south, west)
    x1, y1 = proj.to_xy(north, east)
    xs = np.arange(x0, x1 + spacing_m, spacing_m)
    ys = np.arange(y0, y1 + spacing_m, spacing_m)
    G = nx.MultiDiGraph()
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            lat, lon = proj.to_latlon(x, y)
            G.add_node(i * len(ys) + j, y=float(lat), x=float(lon))
    for i in range(len(xs)):
        for j in range(len(ys)):
            u = i * len(ys) + j
            for v in ([u + len(ys)] if i + 1 < len(xs) else []) + ([u + 1] if j + 1 < len(ys) else []):
                G.add_edge(u, v, length=spacing_m)
                G.add_edge(v, u, length=spacing_m)
    return G


def graph_arrays(G):
    nodes = list(G.nodes)
    idx = {n: k for k, n in enumerate(nodes)}
    lat = np.array([G.nodes[n]["y"] for n in nodes])
    lon = np.array([G.nodes[n]["x"] for n in nodes])
    rows, cols, w = [], [], []
    for u, v, d in G.edges(data=True):
        rows.append(idx[u]); cols.append(idx[v]); w.append(float(d.get("length", 1.0)))
    n = len(nodes)
    # 多重辺は最短のものを採用
    df = pd.DataFrame({"r": rows, "c": cols, "w": w}).groupby(["r", "c"], as_index=False)["w"].min()
    A = csr_matrix((df["w"].to_numpy(), (df["r"].to_numpy(), df["c"].to_numpy())), shape=(n, n))
    return nodes, lat, lon, A


def snap(lat_nodes, lon_nodes, lat_pts, lon_pts, proj: LocalXY):
    """最寄りノードの index と、点→ノードの直線距離（m）。"""
    tree = cKDTree(np.column_stack(proj.to_xy(lat_nodes, lon_nodes)))
    dist, ix = tree.query(np.column_stack(proj.to_xy(lat_pts, lon_pts)))
    return ix, dist


def auto_candidates(lat_n, lon_n, houses, proj: LocalXY, buffer_m: float, spacing_m: float) -> np.ndarray:
    """住宅の凸包＋バッファ内の道路ノードを、最小間隔で間引いて候補とする。"""
    hx, hy = proj.to_xy(houses["lat"].to_numpy(), houses["lon"].to_numpy())
    pts = np.column_stack([hx, hy])
    hull = pts[ConvexHull(pts).vertices]
    nx_, ny_ = proj.to_xy(lat_n, lon_n)
    P = np.column_stack([nx_, ny_])
    inside = np.ones(len(P), dtype=bool)
    # 凸包の各辺から外向きに buffer_m 以内なら内側とみなす（凸多角形のオフセット）
    c = hull.mean(axis=0)
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        e = b - a
        nrm = np.array([e[1], -e[0]]) / np.linalg.norm(e)
        if np.dot(c - a, nrm) > 0:
            nrm = -nrm
        inside &= (P - a) @ nrm <= buffer_m
    chosen: list[int] = []
    for k in np.flatnonzero(inside):
        if all(np.hypot(*(P[k] - P[j])) >= spacing_m for j in chosen):
            chosen.append(k)
    return np.array(chosen, dtype=int)


def shortest(A, sources: np.ndarray) -> np.ndarray:
    return dijkstra(A, directed=True, indices=sources)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["auto", "xml", "overpass", "synthetic"], default="auto")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    out = resolve(cfg["paths"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    houses_csv, is_sample = houses_path(cfg)
    houses = load_houses(houses_csv, cfg)
    print(f"住宅: {len(houses)}戸 ({houses_csv}){' ※サンプル' if is_sample else ''}")
    osm_xml = resolve(cfg["paths"].get("osm_xml") or "data/raw/map.osm")
    if args.source == "auto":
        args.source = "xml" if osm_xml.exists() else "overpass"

    dest_keys = list(cfg["destinations"])
    dports = pd.DataFrame([cfg["destinations"][k]["port"] for k in dest_keys])
    # 施設入口は複数可（近い方を使う）。旧形式の entrance（1点）にも対応
    ent_lists = [cfg["destinations"][k].get("entrances") or [cfg["destinations"][k]["entrance"]] for k in dest_keys]
    dents = pd.DataFrame([e for el in ent_lists for e in el])
    ent_owner = np.array([i for i, el in enumerate(ent_lists) for _ in el])
    all_lat = np.concatenate([houses["lat"], dports["lat"], dents["lat"]])
    all_lon = np.concatenate([houses["lon"], dports["lon"], dents["lon"]])
    proj = LocalXY(all_lat.mean(), all_lon.mean())
    buf = cfg["network"]["bbox_buffer_m"]
    dlat, dlon = buf / proj.ky, buf / proj.kx
    bbox = (all_lon.min() - dlon, all_lat.min() - dlat, all_lon.max() + dlon, all_lat.max() + dlat)

    if args.source == "xml":
        import osmnx as ox

        print(f"道路網: {osm_xml} から作成")
        G_walk, G_bike = graphs_from_xml(osm_xml, cfg)
        ox.save_graphml(G_walk, out / "walk.graphml")
        ox.save_graphml(G_bike, out / "bike.graphml")
        print(f"  徒歩 {G_walk.number_of_nodes()}ノード / 走行 {G_bike.number_of_nodes()}ノード")
    elif args.source == "overpass":
        G_walk = nx.MultiDiGraph(fetch_osm(bbox, "walk", out / "walk.graphml"))
        G_bike = nx.MultiDiGraph(fetch_osm(bbox, "bike", out / "bike.graphml"))
    else:
        print("※ 合成格子ネットワークで計算します（動作確認専用）")
        G_walk = G_bike = synthetic_grid(bbox)

    # --- 徒歩ネットワーク ---
    _, wlat, wlon, Aw = graph_arrays(G_walk)
    Aw = Aw.maximum(Aw.T)  # 徒歩は双方向
    h_ix, h_off = snap(wlat, wlon, houses["lat"].to_numpy(), houses["lon"].to_numpy(), proj)

    cand_csv = resolve(cfg["paths"]["candidates_csv"])
    if cand_csv.exists():
        cdf = pd.read_csv(cand_csv)
        c_ix, c_off = snap(wlat, wlon, cdf["lat"].to_numpy(), cdf["lon"].to_numpy(), proj)
        cand = pd.DataFrame({"lat": cdf["lat"], "lon": cdf["lon"]})
        print(f"候補地点: {cand_csv} から {len(cand)}か所")
    else:
        c_ix = auto_candidates(wlat, wlon, houses, proj,
                               cfg["candidates"]["hull_buffer_m"], cfg["candidates"]["min_spacing_m"])
        c_off = np.zeros(len(c_ix))
        cand = pd.DataFrame({"lat": wlat[c_ix], "lon": wlon[c_ix]})
        print(f"候補地点: 道路ノードから自動抽出 {len(cand)}か所")
    cand.insert(0, "cand_id", np.arange(len(cand)))

    Dw = shortest(Aw, c_ix)                        # (n_cand, n_walk_nodes)
    W = Dw[:, h_ix].T + h_off[:, None] + c_off[None, :]

    pe_ix, pe_off = snap(wlat, wlon, dports["lat"].to_numpy(), dports["lon"].to_numpy(), proj)
    en_ix, en_off = snap(wlat, wlon, dents["lat"].to_numpy(), dents["lon"].to_numpy(), proj)
    De = shortest(Aw, pe_ix)
    E = np.array([min(De[d, en_ix[i]] + en_off[i] for i in np.flatnonzero(ent_owner == d)) + pe_off[d]
                  for d in range(len(dest_keys))])

    # --- 走行ネットワーク（一方通行を考慮し、往路・復路の平均） ---
    _, blat, blon, Ab = graph_arrays(G_bike)
    cb_ix, cb_off = snap(blat, blon, cand["lat"].to_numpy(), cand["lon"].to_numpy(), proj)
    db_ix, db_off = snap(blat, blon, dports["lat"].to_numpy(), dports["lon"].to_numpy(), proj)
    out_d = shortest(Ab, cb_ix)[:, db_ix]
    back_d = shortest(Ab, db_ix)[:, cb_ix].T
    R = (out_d + back_d) / 2 + cb_off[:, None] + db_off[None, :]

    for name, M in (("W", W), ("R", R), ("E", E)):
        if not np.isfinite(M).all():
            bad = int((~np.isfinite(M)).sum())
            raise RuntimeError(f"{name} に到達不能な組が {bad} 件あります。bbox_buffer_m を広げてください。")

    # --- 全徒歩ノード分の行列（アプリで添付されたポート位置を評価するため） ---
    Wn = shortest(Aw, h_ix) + h_off[:, None]                       # (n_houses, n_walk_nodes)
    nb_ix, nb_off = snap(blat, blon, wlat, wlon, proj)              # 徒歩ノード → 最寄り走行ノード
    to_dest = shortest(Ab.T.tocsr(), db_ix).T                       # 走行ノード → 目的地ポート
    from_dest = shortest(Ab, db_ix).T                               # 目的地ポート → 走行ノード
    Rn = (to_dest[nb_ix] + from_dest[nb_ix]) / 2 + nb_off[:, None] + db_off[None, :]
    usable = np.isfinite(Wn).all(0) & np.isfinite(Rn).all(1)
    print(f"  添付ポート評価用: 徒歩ノード {usable.sum()}/{len(usable)} が到達可能")

    np.savez_compressed(out / "matrices.npz", W=W, R=R, E=E,
                        Wn=np.where(np.isfinite(Wn), Wn, -1.0), Rn=np.where(np.isfinite(Rn), Rn, -1.0))
    pd.DataFrame({"lat": wlat, "lon": wlon, "usable": usable}).to_csv(out / "walk_nodes.csv", index=False)
    hs = houses.copy()
    hs["snap_offset_m"] = h_off.round(1)
    hs.to_csv(out / "houses_snapped.csv", index=False)
    cand.to_csv(out / "candidates.csv", index=False)
    meta = {
        "source": args.source,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dest_keys": dest_keys,
        "houses_file": str(houses_csv),
        "is_sample": is_sample,
        "input_hash": file_hash(houses_csv),
        "n_houses": len(houses), "n_candidates": len(cand),
        "walk_nodes": G_walk.number_of_nodes(), "bike_nodes": G_bike.number_of_nodes(),
        "bbox": bbox,
    }
    if args.source == "xml":
        meta["osm_xml"] = str(osm_xml)
        meta["osm_xml_hash"] = file_hash(osm_xml)
    if args.source in ("xml", "overpass"):
        import osmnx as ox
        meta["osmnx_version"] = ox.__version__
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"保存しました: {out}")
    ride_mean = {k: round(float(v)) for k, v in zip(dest_keys, R.mean(0))}
    print(f"  徒歩 住宅→候補: 平均 {W.mean():.0f} m / 走行 候補→目的地ポート: 平均 {ride_mean} m")


if __name__ == "__main__":
    main()
