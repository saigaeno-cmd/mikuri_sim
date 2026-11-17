"""修士論文用の比較ケースをまとめて計算し、1ケース1ページの結果シートと、比較の表・曲線の PPT を作る。

  python scripts/run_thesis_cases.py                 # 全ケース × 5回（乱数を変える）
  python scripts/run_thesis_cases.py --reps 10       # 回数を変える
  python scripts/run_thesis_cases.py --only S1 A2    # 一部のケースだけ
  python scripts/run_thesis_cases.py --render-only   # 計算済み（キャッシュ）から図だけ作り直す
ケースは data/conditions/thesis_cases.yaml。出力は data/output/thesis/：
  修士論文_比較.pptx（条件・比較の表・曲線・各ケース1ページ）、slides/*.png、比較.csv、全回.csv
"""
from __future__ import annotations

import argparse
import hashlib
import io
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import opt_multi as om  # noqa: E402
from src import sheet, thesis  # noqa: E402
from src.loaders import load_config, load_network, load_survey, resolve, survey_path  # noqa: E402

G = {}


def _init():
    cfg = load_config()
    net = load_network(cfg)
    cfg["simulation"]["n_households"] = cfg["simulation"].get("n_households") or len(net.houses)
    G.update(cfg=cfg, net=net, survey=load_survey(survey_path(cfg)[0], cfg))


def _key(p, seed):
    h = hashlib.md5(repr((sorted(p.items()), seed, (ROOT / "config.yaml").read_bytes(),
                          (ROOT / "data" / "processed" / "meta.json").read_bytes())).encode()).hexdigest()
    return h[:16]


def job(args):
    cid, p, seed, cache_dir = args
    f = Path(cache_dir) / f"{cid}_{seed}_{_key(p, seed)}.pkl"
    if f.exists():
        return cid, seed, pickle.loads(f.read_bytes())
    if not G:
        _init()
    t0 = time.time()
    R = thesis.run_case(G["net"], G["cfg"], G["survey"], p, seed=seed)
    R["_sec"] = time.time() - t0
    if seed != 0:                                  # 2回目以降は表の値だけ残す（容量の節約）
        R = {"_summary": thesis.summary(R), "_sec": R["_sec"]}
    f.write_bytes(pickle.dumps(R))
    return cid, seed, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(ROOT / "data" / "conditions" / "thesis_cases.yaml"))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--render-only", action="store_true")
    a = ap.parse_args()
    spec = yaml.safe_load(Path(a.cases).read_text(encoding="utf-8"))
    base, cases = spec["base"], spec["cases"]
    if a.only:
        cases = [c for c in cases if c["id"] in a.only]
    cfg = load_config()
    out = resolve(cfg["paths"].get("output_dir", "data/output")) / "thesis"
    (out / "slides").mkdir(parents=True, exist_ok=True)
    cache = out / ".cache"
    cache.mkdir(exist_ok=True)
    jobs = [(c["id"], thesis.case_params(base, c), s, str(cache)) for c in cases for s in range(a.reps)]
    t0 = time.time()
    res = {}
    if a.render_only:
        for j in jobs:
            f = cache / f"{j[0]}_{j[2]}_{_key(j[1], j[2])}.pkl"
            if f.exists():
                res[(j[0], j[2])] = pickle.loads(f.read_bytes())
    else:
        print(f"{len(jobs)} 件（{len(cases)}ケース × {a.reps}回）を計算します")
        with Pool(a.procs, initializer=_init) as pool:
            for i, (cid, seed, R) in enumerate(pool.imap_unordered(job, jobs), 1):
                res[(cid, seed)] = R
                print(f"  {i}/{len(jobs)} {cid} 回{seed}  {R.get('_sec', 0):.0f}秒  （経過 {time.time() - t0:.0f}秒）", flush=True)
    _init()
    cfg, net = G["cfg"], G["net"]
    for R in res.values():
        if "L" in R:
            R["_houses"] = net.houses
    # ---- 表 ----
    rows = []
    for c in cases:
        for s in range(a.reps):
            R = res.get((c["id"], s))
            if R is None:
                continue
            sm = R["_summary"] if "_summary" in R else thesis.summary(R)
            rows.append({"ID": c["id"], "比較": c["group"], "ケース": c["name"], "回": s, **sm})
    allruns = pd.DataFrame(rows)
    allruns.to_csv(out / "全回.csv", index=False, encoding="utf-8-sig")
    comp, stats = [], {}
    for c in cases:
        g = allruns[allruns["ID"] == c["id"]]
        if not len(g) or "エラー" in g and g["エラー"].notna().all():
            continue
        p = thesis.case_params(base, c)
        ok = g["達成"].astype(bool)
        st_ = {k: thesis._fmt_range(g[k].tolist()) for k in ("合計台数", "自転車", "キックボード", "4輪", "駅ポート必要容量")}
        st_["充足率"] = thesis._fmt_range(g["充足率"].tolist(), pct=True)
        st_["平均短縮移動時間"] = thesis._fmt_range(g["平均短縮移動時間"].tolist(), fmt="{:.1f}分")
        st_ = {k: (f"{a.reps}回：{v}" if v != "—" else "") for k, v in st_.items()}
        stats[c["id"]] = st_
        comp.append({
            "ID": c["id"], "比較": c["group"], "ケース": c["name"],
            "需要": f"①時間短縮（{p['A']:g}分・{p['X']:g}分）" if p["model"] == "time" else f"転換率{p['conversion']:.0%}",
            "ポート数": thesis._fmt_range(g["ポート数"].tolist()),
            "台数": (thesis._fmt_range(g.loc[ok, "合計台数"].tolist()) if ok.all() else
                    f"{len(g)}回中{ok.sum()}回達成" if ok.any() else "達成不可"),
            "内訳 自・キ・4輪": "・".join(f"{np.median(g.loc[ok if ok.any() else ~ok, k]):.0f}"
                                     for k in ("自転車", "キックボード", "4輪")),
            "充足率": thesis._fmt_range(g["充足率"].tolist(), pct=True),
            "上限の目安": thesis._fmt_range(g["最大充足率"].tolist(), pct=True),
            "駅必要容量": thesis._fmt_range(g["駅ポート必要容量"].tolist()),
            "平均短縮(分)": thesis._fmt_range(g["平均短縮移動時間"].tolist(), fmt="{:.1f}"),
            "最大短縮(分)": thesis._fmt_range(g["最大短縮時間"].tolist(), fmt="{:.1f}"),
            "最小短縮(分)": thesis._fmt_range(g["最小短縮時間"].tolist(), fmt="{:.1f}"),
            "平均移動(分)": thesis._fmt_range(g["平均移動時間"].tolist(), fmt="{:.1f}"),
            "最大移動(分)": thesis._fmt_range(g["最大移動時間"].tolist(), fmt="{:.1f}"),
            "最小移動(分)": thesis._fmt_range(g["最小移動時間"].tolist(), fmt="{:.1f}"),
            "徒歩だけ平均(分)": thesis._fmt_range(g["徒歩だけの平均"].tolist(), fmt="{:.1f}"),
            "年間費用(万円)": thesis._fmt_range(g["年間総費用(万円)"].tolist()),
            "1回あたり(円)": thesis._fmt_range(g["1回あたり費用(円)"].tolist()),
        })
    comp = pd.DataFrame(comp)
    comp.to_csv(out / "比較.csv", index=False, encoding="utf-8-sig")
    print(comp.to_string())
    # ---- スライド ----
    slides = []
    tgt = float(base["target"])
    oc = cfg["opt_time"]
    fixed = pd.DataFrame([
        ("対象", f"既存街区171戸＋新規街区59戸（{len(net.houses)}戸）"),
        ("目的地", "南栗橋駅・イオンスタイル・公園・エニタイム（ご指定の座標）"),
        ("需要", "街区230戸の時間帯別・目的地別の移動需要（平日推計・公的統計ベース）× 転換率。休日も同じ表"),
        ("車種と用途", "1 住宅⇔駅：キ・4輪／2 駅⇔イオン：4輪・自／3 住宅⇔イオン：4輪・自／4 駅⇔公園：自／5 住宅⇔エニタイム：3車種／6 住宅⇔公園：自"),
        ("運用", f"夜間に1回、初期配置に戻す／手続き{oc['procedure_sec']}秒／走行速度は経路別の実測値"),
        ("期間・天気", f"{'・'.join(f'{m}月' for m in oc['months'])}の3か月（平日・休日・天気は平年値から抽選）"),
        ("基本ケース", f"転換率{base['conversion']:.0%}・ポート上限{base['k_max']}か所（候補地{len(net.candidates)}か所から選ぶ）・"
                    f"住宅側ポート容量{base['port_cap']}台・駅ポート{base['dest_caps']['station']}台・目標充足率{tgt:.0%}・3車種"),
        ("利用しない条件", f"進行方向と反対方向のポートへ{cfg['behavior'].get('backtrack_limit_min')}分以上歩く場合は利用しない"
                       "（住宅から見てポートの方向と目的地の方向のなす角が90度を超える）"),
        ("繰り返し", f"各ケース {a.reps}回（乱数を変える）。表は中央値（最小–最大）"),
    ], columns=["項目", "条件"])
    slides.append(thesis.table_slide("シミュレーションの条件（固定）", fixed, col_w=[0.16, 0.84], fs=12.5))
    vary = pd.DataFrame([(c["group"], c["id"], c["name"]) for c in cases], columns=["比較", "ID", "変える条件（基本ケースから1つだけ）"])
    slides.append(thesis.table_slide("比べるケース", vary, col_w=[0.2, 0.1, 0.7], fs=12.5))
    if len(comp):
        tcols = ["平均短縮(分)", "最大短縮(分)", "最小短縮(分)", "平均移動(分)", "最大移動(分)", "最小移動(分)", "徒歩だけ平均(分)"]
        show = comp.drop(columns=["比較"] + tcols)
        w = [0.04, 0.19, 0.12, 0.06, 0.1, 0.09, 0.09, 0.09, 0.08, 0.1, 0.1]
        slides.append(thesis.table_slide("比較の表（台数・充足率・費用）", show, sub=f"各ケース{a.reps}回の中央値（最小–最大）。"
                                         f"台数＝目標充足率{tgt:.0%}を満たす最小台数（達成不可は容量まで埋めた参考の内訳）。"
                                         "上限の目安＝すべてのポートを容量まで埋めたときの充足率",
                                         col_w=w, fs=9, highlight=lambda i: show["ID"].iloc[i] == "S1"))
        show2 = comp[["ID", "ケース", "需要"] + tcols]
        w2 = [0.04, 0.2, 0.13] + [0.09] * 7
        slides.append(thesis.table_slide("比較の表（移動時間・片道）", show2, sub=f"各ケース{a.reps}回の中央値（最小–最大）。"
                                         "短縮＝徒歩だけの時間 − 実際の時間（利用できなかった人は徒歩、マイナス＝徒歩より遅い）。"
                                         "最大・最小は3か月の全片道のうちの最大・最小",
                                         col_w=w2, fs=9, highlight=lambda i: show2["ID"].iloc[i] == "S1"))
    curves = {}
    for cid, lab in (("S1", "低位12%"), ("A2", "高位24%")):
        R = res.get((cid, 0))
        if R and "L" in R:
            curves[lab] = om.fleet_curve(R, spec.get("curve_sizes", [10, 20, 30, 40, 50, 60]))
    if curves:
        slides.append(thesis.curve_slide(curves, tgt, sub="基本ケース（S1）と高位（A2）で選ばれた配置のまま、台数だけを変えた（1回目の乱数）。"
                                         "右端は、住宅側・駅・目的地のポートの区画がすべて埋まる台数"))
    bids = [c for c in cases if c["group"].startswith("B") or c["id"] == "S1"]
    if len(bids) > 2:
        bd = []
        for c in bids:
            g = allruns[allruns["ID"] == c["id"]]
            if len(g):
                p = thesis.case_params(base, c)
                ok = g["達成"].astype(bool)
                hit = bool(ok.mean() >= 0.5)            # 半数以上の回で目標を達成
                bd.append({"ポート数上限": int(p["k_max"]), "達成": hit, "達成回": f"{int(ok.sum())}/{len(g)}",
                           "合計台数": float(np.median(g.loc[ok if hit else g.index, "合計台数"])),
                           "最大充足率": float(np.median(g["最大充足率"]))})
        bd = pd.DataFrame(bd).sort_values("ポート数上限")
        slides.append(thesis.ports_slide(bd, tgt, sub=f"転換率{base['conversion']:.0%}・住宅側ポート容量{base['port_cap']}台。各{a.reps}回の中央値"))
    if spec.get("heatmap"):
        hm = thesis.heatmap(net, cfg, {**base, **spec["heatmap"]})
        hm.to_csv(out / "ヒートマップ.csv", index=False, encoding="utf-8-sig")
        slides.append(thesis.heatmap_slide(hm, sub="ポート3か所・目的地4か所・3車種。数字＝条件を満たす住宅の数"))
    roads = sheet._roads(resolve(cfg["paths"]["processed_dir"]))
    for c in cases:
        R = res.get((c["id"], 0))
        if not R or "L" not in R:
            continue
        p = thesis.case_params(base, c)
        cond = (f"{c['group']}／{c['name']}：" +
                (f"①時間短縮型（ポートまで徒歩{p['A']:g}分以内・{p['X']:g}分以上短縮）" if p["model"] == "time"
                 else f"②候補地から選ぶ（需要 転換率{p['conversion']:.0%}）")
                + f"・ポート上限{p['k_max']}か所・住宅側容量{p['port_cap']}台・駅ポート"
                + ("上限なし" if p["dest_caps"]["station"] >= 900 else f"{p['dest_caps']['station']}台")
                + f"・目標{p['target']:.0%}\n車種：{'・'.join(om.TYPE_LABEL[v] for v in p['vehicles'])}／"
                f"9〜11月の3か月／図は1回目の結果、下の小さい数字は{a.reps}回の中央値（最小–最大）")
        slides.append(thesis.case_sheet(R, cfg, f"{c['id']}　{c['name']}", cond, stats.get(c["id"]), roads))
    pngs = []
    for i, fig in enumerate(slides, 1):
        b = sheet.to_png(fig, dpi=150)
        (out / "slides" / f"{i:02d}.png").write_bytes(b)
        pngs.append(b)
    (out / "修士論文_比較.pptx").write_bytes(sheet.to_pptx(pngs))
    from PIL import Image
    ims = [Image.open(io.BytesIO(b)).convert("RGB") for b in pngs]
    ims[0].save(out / "修士論文_比較.pdf", save_all=True, append_images=ims[1:])
    print(f"保存：{out}（{len(pngs)}枚、{time.time() - t0:.0f}秒）")


if __name__ == "__main__":
    main()
