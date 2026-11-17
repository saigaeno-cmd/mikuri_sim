"""座標計算の小道具（狭い範囲なので局所平面近似で十分）。"""
from __future__ import annotations

import numpy as np

EARTH_R = 6_371_000.0


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


class LocalXY:
    """緯度経度 ⇄ 原点からのメートル座標。"""

    def __init__(self, lat0: float, lon0: float):
        self.lat0, self.lon0 = lat0, lon0
        self.kx = np.radians(1) * EARTH_R * np.cos(np.radians(lat0))
        self.ky = np.radians(1) * EARTH_R

    def to_xy(self, lat, lon):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky

    def to_latlon(self, x, y):
        return self.lat0 + np.asarray(y) / self.ky, self.lon0 + np.asarray(x) / self.kx


def backtrack_mask(house_lat, house_lon, port_lat, port_lon, dest_lat, dest_lon, walk_min, limit_min):
    """「進行方向と反対方向のポートへ limit_min 分以上歩く」組（住宅, ポート）＝利用しない、を True で返す。
    反対方向：住宅から見て、ポートの方向と目的地の方向のなす角が90度を超える（直線の向きで判定）。
    walk_min：(住宅, ポート) 住宅→ポートの徒歩（分）。limit_min が None なら判定しない（すべて False）。"""
    walk_min = np.asarray(walk_min, float)
    if limit_min is None:
        return np.zeros(walk_min.shape, bool)
    lat0, lon0 = float(np.mean(house_lat)), float(np.mean(house_lon))
    xy = LocalXY(lat0, lon0)
    hx, hy = xy.to_xy(np.asarray(house_lat, float), np.asarray(house_lon, float))
    px, py = xy.to_xy(np.asarray(port_lat, float), np.asarray(port_lon, float))
    dx, dy = xy.to_xy(dest_lat, dest_lon)
    dot = (px[None, :] - hx[:, None]) * (dx - hx)[:, None] + (py[None, :] - hy[:, None]) * (dy - hy)[:, None]
    return (dot < 0) & (walk_min >= float(limit_min))


def dest_point(cfg, key) -> tuple:
    """目的地 key の代表点（最初の入口、無ければポート）。"""
    d = cfg["destinations"][key]
    e = (d.get("entrances") or [d.get("entrance") or d["port"]])[0]
    return float(e["lat"]), float(e["lon"])


def backtrack_limit(cfg):
    """設定の「反対方向に歩く」上限（分）。None なら判定しない。"""
    return (cfg.get("behavior") or {}).get("backtrack_limit_min")
