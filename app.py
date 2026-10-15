"""南栗橋BLP街区 モビリティシェア シミュレーター（streamlit run app.py）

ページ
- 感度分析：転換率・ポート数ごとの必要台数とポート位置の感度、配置ファイルの添付
- ① ポート最適化：需要・気象条件から、移動時間最短・充足率最大のポートと必要台数
- ② ポート配置評価：地図でポートを選び、台数を指定して合計移動時間と充足率を算出
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="南栗橋BLP モビリティシェア", page_icon="🚲", layout="wide")
nav = st.navigation([
    st.Page("views/sensitivity.py", title="感度分析", icon="📈", default=True),
    st.Page("views/optimize.py", title="① ポート最適化", icon="🎯"),
    st.Page("views/evaluate.py", title="② ポート配置評価", icon="🗺️"),
])
nav.run()
