"""設定ページ：需要シナリオ・費用単価・駅ポート・運用ルール・街区を画面から変更して config.yaml に保存する。"""
from __future__ import annotations

import datetime as dt
import io
import re
import shutil
import subprocess
import sys

import pandas as pd
import streamlit as st
from ruamel.yaml import YAML

from src.loaders import ROOT, load_houses

CFG_PATH = ROOT / "config.yaml"
BACKUP = ROOT / "data" / "backup"


def _yaml():
    y = YAML()
    y.preserve_quotes = True
    y.width = 4096
    y.indent(mapping=2, sequence=4, offset=2)
    y.representer.add_representer(type(None), lambda r, d: r.represent_scalar("tag:yaml.org,2002:null", "null"))
    return y


def load_doc():
    return _yaml().load(CFG_PATH.read_text(encoding="utf-8"))


def save_doc(doc, note: str):
    """保存前に data/backup/ へ元の config.yaml を残す。"""
    BACKUP.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy(CFG_PATH, BACKUP / f"config_{stamp}.yaml")
    buf = io.StringIO()
    _yaml().dump(doc, buf)
    CFG_PATH.write_text(buf.getvalue(), encoding="utf-8")
    st.cache_resource.clear()
    st.cache_data.clear()
    st.success(f"保存しました（{note}）。元の設定は data/backup/config_{stamp}.yaml にあります。")


def precompute():
    with st.spinner("道路網と距離を計算し直しています（15秒ほど）…"):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "precompute_network.py")], cwd=ROOT,
                           capture_output=True, text=True)
    st.cache_resource.clear()
    st.cache_data.clear()
    if r.returncode == 0:
        st.success("前処理が終わりました。")
        st.code(r.stdout[-1500:])
    else:
        st.error("前処理に失敗しました。")
        st.code((r.stdout + r.stderr)[-3000:])


st.title("設定")
st.caption("ここで変えた値は config.yaml に保存され、画面とバッチ計算（scripts/run_all.py）の両方に反映されます。"
           "全条件の計算結果と報告資料は、保存後に `python scripts/run_all.py` を実行すると更新されます。")
doc = load_doc()
t1, t2, t3, t4, t5 = st.tabs(["需要シナリオ", "費用単価", "駅ポート", "運用・行動", "街区・道路"])

# ---------- 需要 ----------
with t1:
    st.markdown("転換率＝アンケートで「徒歩」「徒歩＋電車」と答えた移動のうち、モビリティに切り替える割合。")
    ds = pd.DataFrame({"名前": list(doc["demand_scenarios"].keys()),
                       "転換率（%）": [float(v) * 100 for v in doc["demand_scenarios"].values()]})
    ed = st.data_editor(ds, num_rows="dynamic", hide_index=True, use_container_width=True, key="set_dem",
                        column_config={"転換率（%）": st.column_config.NumberColumn(min_value=0.0, max_value=100.0,
                                                                                    step=1.0, format="%.1f")})
    st.caption("名前の先頭の2つが報告資料の「低位」「高位」になります（例：低位（暫定）→ 低位（確定））。")
    if st.button("需要シナリオを保存", key="save_dem"):
        ed = ed.dropna()
        if ed.empty or ed["名前"].duplicated().any():
            st.error("名前が空か重複しています。")
        else:
            doc["demand_scenarios"].clear()
            for n, v in zip(ed["名前"], ed["転換率（%）"]):
                doc["demand_scenarios"][str(n)] = round(float(v) / 100, 4)
            save_doc(doc, "需要シナリオ")

# ---------- 費用 ----------
with t2:
    c = doc["cost"]
    a, b = st.columns(2)
    v = {}
    v["eb"] = a.number_input("電動アシスト自転車 単価（円/台）", 0, 2_000_000, int(c["price"]["ebike"]), 10000)
    v["kb"] = b.number_input("電動キックボード 単価（円/台）", 0, 2_000_000, int(c["price"]["kickboard"]), 10000)
    v["life"] = a.number_input("車両の償却年数（年）", 1, 15, int(c["life_years"]))
    v["mnt"] = b.number_input("車両の維持費（円/台/年）", 0, 500_000, int(c["maintenance"]), 1000)
    v["port"] = a.number_input("ポート初期費用（円/か所）", 0, 10_000_000, int(c["port_initial"]), 10000)
    v["plife"] = b.number_input("ポートの償却年数（年）", 1, 30, int(c["port_life_years"]))
    v["land"] = a.number_input("用地（円/区画/月）", 0, 20000, int(c["land_per_slot_month"]), 50)
    v["labor"] = b.number_input("人件費（円/時）", 0, 10000, int(c["labor_per_hour"]), 50)
    v["truck"] = a.number_input("軽トラック（円/回）", 0, 50000, int(c["truck_per_op"]), 50)
    v["swap"] = b.number_input("電池交換の作業（分/回）", 0, 120,
                               int(doc["battery"]["kickboard"].get("swap_work_min", 0) or 0))
    if st.button("費用単価を保存", key="save_cost"):
        c["price"]["ebike"], c["price"]["kickboard"] = int(v["eb"]), int(v["kb"])
        c["life_years"], c["maintenance"] = int(v["life"]), int(v["mnt"])
        c["port_initial"], c["port_life_years"] = int(v["port"]), int(v["plife"])
        c["land_per_slot_month"], c["labor_per_hour"], c["truck_per_op"] = int(v["land"]), int(v["labor"]), int(v["truck"])
        doc["battery"]["kickboard"]["swap_work_min"] = int(v["swap"])
        save_doc(doc, "費用単価")

# ---------- 駅ポート ----------
with t3:
    sp = doc["station_port"]
    base = st.number_input("駅ポートの基本の容量（台）", 1, 200, int(sp["max_slots"]))
    cc = pd.DataFrame({"名前": list(sp["capacity_cases"].keys()), "容量（台）": list(sp["capacity_cases"].values())})
    st.markdown("全条件の計算で比べる容量（区画の案）")
    cce = st.data_editor(cc, num_rows="dynamic", hide_index=True, use_container_width=True, key="set_cap")
    st.caption("1台あたり 1m×2m。案を増やすと全条件の計算時間がその分増えます。")
    if st.button("駅ポートを保存", key="save_cap"):
        cce = cce.dropna()
        sp["max_slots"] = int(base)
        sp["capacity_cases"].clear()
        for n, v_ in zip(cce["名前"], cce["容量（台）"]):
            sp["capacity_cases"][str(n)] = int(v_)
        save_doc(doc, "駅ポート")

# ---------- 運用・行動 ----------
with t4:
    fm, rb = doc["fleet_mix"], doc["rebalance"]
    sub = st.radio("希望車種が無いとき", ["(ii) 他の車種があれば乗り、無ければ徒歩", "(i) 徒歩"],
                   index=0 if fm["substitution"] else 1)
    es = st.slider("電動アシスト自転車を希望する人の割合（%）", 0, 100, int(round(float(fm["ebike_share"]) * 100)))
    fr = st.radio("満車で返せない車両の扱い", ["区画外に置き、次の作業で回収", "運営者が回収に出る"],
                  index=0 if rb.get("full_return", "outside") == "outside" else 1)
    cd = st.number_input("運営者が回収に出るまで（分）", 5, 600, int(rb.get("collect_delay_min", 60)), 5)
    beh = doc.get("behavior") or {}
    al = st.slider("利用離れの強さ α（0＝なし）", 0.0, 2.0, float(beh.get("discourage_alpha", 0) or 0), 0.1,
                   help="前日に借りられなかった割合 f に対し、翌日の利用希望を (1−α·f) 倍にする。")
    ps = st.number_input("手続き時間（秒/回）", 0, 600, int(doc["procedure_sec"]), 10)
    bk = doc["battery"]["kickboard"]
    st.markdown("**電動キックボードの電池**（導入車両のカタログ値に置き換える）")
    b1, b2, b3 = st.columns(3)
    rng_km = b1.number_input("航続距離（km）", 1, 200, int(bk["range_km"]))
    below = b2.number_input("交換する残量（%）", 0, 90, int(round(float(bk["swap_below"]) * 100)))
    outh = b3.number_input("交換・充電で使えない時間（時間）", 0.0, 24.0, float(bk["out_hours"]), 0.5)
    if st.button("運用・行動を保存", key="save_ops"):
        fm["substitution"] = sub.startswith("(ii)")
        fm["ebike_share"] = round(es / 100, 3)
        rb["full_return"] = "outside" if fr.startswith("区画外") else "collect"
        rb["collect_delay_min"] = int(cd)
        if "behavior" not in doc:
            doc["behavior"] = {}
        doc["behavior"]["discourage_alpha"] = float(al)
        doc["procedure_sec"] = int(ps)
        bk["range_km"], bk["swap_below"], bk["out_hours"] = int(rng_km), round(below / 100, 3), float(outh)
        save_doc(doc, "運用・行動")

# ---------- 街区・道路 ----------
with t5:
    hs = doc["paths"]["houses"]
    items = [{"name": "街区", "file": hs}] if isinstance(hs, str) else [dict(d) for d in hs]
    st.markdown("**登録されている街区**")
    rows = []
    for it in items:
        p = ROOT / it["file"]
        try:
            n = len(load_houses(p, doc)) if p.exists() else "ファイルなし"
        except Exception as e:  # noqa: BLE001
            n = f"読込エラー：{e}"
        rows.append({"街区": it["name"], "ファイル": it["file"], "戸数": n})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("**街区を追加**（点データ：GeoPackage・GeoJSON、または lat・lon 列のCSV）")
    a, b = st.columns([1, 2])
    name = a.text_input("街区の名前", placeholder="例：第3街区")
    up = b.file_uploader("住宅・区画のファイル", type=["gpkg", "geojson", "json", "csv"], key="set_up")
    if st.button("追加して前処理を実行", disabled=not (name and up), key="add_dist"):
        if any(it["name"] == name for it in items):
            st.error("同じ名前の街区があります。")
        else:
            safe = re.sub(r"[^\w\-]", "_", name)
            dst = ROOT / "data" / "raw" / f"houses_{safe}{'.' + up.name.rsplit('.', 1)[-1].lower()}"
            dst.write_bytes(up.getvalue())
            try:
                n = len(load_houses(dst, doc))
            except Exception as e:  # noqa: BLE001
                dst.unlink(missing_ok=True)
                st.error(f"ファイルを読めませんでした：{e}")
                st.stop()
            new = {"name": name, "file": str(dst.relative_to(ROOT))}
            if isinstance(hs, str):
                doc["paths"]["houses"] = [{"name": "街区", "file": hs}, new]
            else:
                hs.append(new)
            save_doc(doc, f"街区「{name}」{n}戸を追加")
            precompute()
    if len(items) > 1:
        rm = st.selectbox("登録から外す街区", ["（選択）"] + [it["name"] for it in items])
        if rm != "（選択）" and st.button(f"「{rm}」を外して前処理を実行", key="rm_dist"):
            doc["paths"]["houses"] = [d for d in doc["paths"]["houses"] if d["name"] != rm]
            save_doc(doc, f"街区「{rm}」を外す（ファイルは残す）")
            precompute()

    st.markdown("**道路を追加**（新しい区画道路などを、OSM XML の部分ファイルで足し込む）")
    osm = st.file_uploader("追加する道路（.osm）", type=["osm", "xml"], key="set_osm")
    if st.button("道路を統合して前処理を実行", disabled=osm is None, key="add_osm"):
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        patch = ROOT / "data" / "raw" / f"roads_patch_{stamp}.osm"
        patch.write_bytes(osm.getvalue())
        base_osm = ROOT / doc["paths"]["osm_xml"]
        out = ROOT / "data" / "raw" / f"map_merged_{stamp}.osm"
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "merge_osm.py"), str(base_osm), str(patch), str(out)],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            st.error("統合に失敗しました。")
            st.code((r.stdout + r.stderr)[-2000:])
        else:
            st.info(r.stdout)
            doc["paths"]["osm_xml"] = str(out.relative_to(ROOT))
            save_doc(doc, "道路を統合")
            precompute()
    if st.button("前処理だけ実行", key="pre_only"):
        precompute()
