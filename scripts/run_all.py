"""一括更新：道路網の前処理 → 全条件の計算 → 感度分析 → 追加検証 → 報告資料。

需要（config.yaml の demand_scenarios）や街区・道路を変えたら、これを1回実行すれば結果と報告資料がそろう。
使い方:
    python scripts/run_all.py                 # すべて（4コアで1時間前後）
    python scripts/run_all.py --skip-precompute
    python scripts/run_all.py --only report   # 報告資料だけ作り直す
    python scripts/run_all.py --quick         # 動作確認用（条件と回数を減らす。報告資料は作らない）
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STEPS = ["precompute", "study", "sensitivity", "checks", "report"]


def run(name, args):
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / args[0]), *args[1:]], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"{name} で失敗しました（終了コード {r.returncode}）。")
    print(f"--- {name}：{time.time() - t0:.0f}秒", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--districts", nargs="*", default=None, help="対象街区。省略で全街区")
    ap.add_argument("--skip-precompute", action="store_true")
    ap.add_argument("--only", nargs="*", choices=STEPS, help="指定した工程だけ実行")
    ap.add_argument("--quick", action="store_true", help="回数を減らして動作確認")
    a = ap.parse_args()
    steps = a.only or [s for s in STEPS if not (s == "precompute" and a.skip_precompute)]
    dist = ["--districts", *a.districts] if a.districts else []
    procs = ["--procs", str(a.procs)]
    reps = "2" if a.quick else "10"
    cmds = {
        "precompute": ["precompute_network.py"],
        "study": ["run_study.py", "--reps", reps, *procs, *dist, *(["--quick"] if a.quick else [])],
        "sensitivity": ["run_sensitivity.py", "--reps", "2" if a.quick else "10", *procs, *dist],
        "checks": ["run_checks.py", "--reps", "1" if a.quick else "3", *procs, *dist],
        "report": ["make_report.py"],
    }
    if a.quick and "report" in steps and not a.only:
        steps.remove("report")
        print("--quick では条件を絞るため、報告資料は作りません（全条件の結果が必要）。")
    t0 = time.time()
    for s in steps:
        run(s, cmds[s])
    print(f"\n完了（{(time.time() - t0) / 60:.0f}分）。報告資料：data/output/study/報告_モビリティシェア検討.pptx")


if __name__ == "__main__":
    main()
