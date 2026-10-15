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
