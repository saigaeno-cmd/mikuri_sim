"""南栗橋BLP街区 モビリティシェア シミュレーター（streamlit run app.py）

ページ
- 感度分析：転換率・ポート数ごとの必要台数とポート位置の感度、配置ファイルの添付
- ① ポート最適化（時間短縮）：徒歩より〇分以上短くなる住宅が最も多いポートと、必要台数
- ② ポート配置評価：地図で候補地と配置可能台数を選んで評価。候補地から充足率が最大の組み合わせも探す
- ③ アンケート需要の最適化：アンケートを需要の上限（転換率100%）とし、転換率を指定して最適ポート・台数
- 実証（台数固定）：10台などの限られた台数で、転換率ごとの充足率・ポート数・車種の割合
- 計算結果：全条件×10回・感度分析・追加検証の結果を条件で絞って表示
- 発表スライド：修論発表などで投影する6画面（16:9）。PNG・PPTで保存
- 設定：需要シナリオ・費用単価・駅ポート・運用ルール・街区・道路を画面から変更
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="南栗橋BLP モビリティシェア", page_icon="🚲", layout="wide")
nav = st.navigation([
    st.Page("views/sensitivity.py", title="感度分析", icon="📈", default=True),
    st.Page("views/optimize_time.py", title="① ポート最適化（時間短縮）", icon="⏱️", url_path="optimize_time"),
    st.Page("views/evaluate.py", title="② ポート配置評価", icon="🗺️"),
    st.Page("views/optimize.py", title="③ アンケート需要の最適化", icon="🎯", url_path="optimize"),
    st.Page("views/pilot.py", title="実証（台数固定）", icon="🧪"),
    st.Page("views/results.py", title="計算結果", icon="📊"),
    st.Page("views/present.py", title="発表スライド", icon="🎤"),
    st.Page("views/settings.py", title="設定", icon="⚙️"),
])
nav.run()
