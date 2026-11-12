"""②ポート配置評価の候補地入力マップ（Leaflet を同梱した小さな Streamlit コンポーネント）。
ダブルクリックで候補地を追加し、容量（台）を入力。候補地をクリックすると容量の変更・CANCEL（取り消し）ができる。"""
from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_COMP = components.declare_component("portmap", path=str(Path(__file__).parent / "components" / "portmap"))


def portmap(*, houses, dests, cands, ver: int, center, zoom: int = 17, height: int = 520, default_cap: int = 5,
            key=None):
    """戻り値：{"cands": [{name, lat, lon, cap}], "nonce": 送信番号, "ver": 版} か None（まだ操作なし）。
    ver を変えると、地図上の候補地を cands で置き換える（CLEAR など）。"""
    return _COMP(houses=houses, dests=dests, cands=cands, ver=ver, center=center, zoom=zoom, height=height,
                 default_cap=default_cap, key=key, default=None)
