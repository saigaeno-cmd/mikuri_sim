"""OSM XML に追加道路（部分抽出の OSM XML）を足し込む。

追加側にしかないノード・way を加え、両方にある way は追加側のノード列で置き換える（接続点の差し込み）。
既存ノードの座標が食い違う場合は止める。
使い方: python scripts/merge_osm.py data/raw/map.osm data/raw/new_roads_patch.osm data/raw/map_merged.osm
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET


def main(base_p, patch_p, out_p):
    base = ET.parse(base_p)
    r0, r1 = base.getroot(), ET.parse(patch_p).getroot()
    n0 = {n.get("id"): n for n in r0.iter("node")}
    w0 = {w.get("id"): w for w in r0.iter("way")}
    bad = [n.get("id") for n in r1.iter("node")
           if n.get("id") in n0 and (n.get("lat"), n.get("lon")) != (n0[n.get("id")].get("lat"), n0[n.get("id")].get("lon"))]
    if bad:
        sys.exit(f"既存ノードの座標が異なります：{bad[:5]}")
    kids = list(r0)
    last_node = max(i for i, e in enumerate(kids) if e.tag == "node")
    new_nodes = [n for n in r1.iter("node") if n.get("id") not in n0]
    for k, n in enumerate(new_nodes):
        r0.insert(last_node + 1 + k, n)
    kids = list(r0)
    last_way = max(i for i, e in enumerate(kids) if e.tag == "way")
    new_ways, replaced = [], 0
    for w in r1.iter("way"):
        if w.get("id") in w0:
            old = w0[w.get("id")]
            if [nd.get("ref") for nd in old.iter("nd")] != [nd.get("ref") for nd in w.iter("nd")]:
                for nd in list(old.iter("nd")):
                    old.remove(nd)
                for i, nd in enumerate(w.iter("nd")):
                    old.insert(i, nd)
                replaced += 1
        else:
            new_ways.append(w)
    for k, w in enumerate(new_ways):
        r0.insert(last_way + 1 + k, w)
    base.write(out_p, encoding="utf-8", xml_declaration=True)
    print(f"追加ノード {len(new_nodes)}・追加way {len(new_ways)}・接続点を差し込んだway {replaced} → {out_p}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
