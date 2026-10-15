@echo off
chcp 65001 > nul
cd /d %~dp0
rem 気になるワード：サーバーを起動し、リンク一覧（管理者画面・投影画面・スマホ回答）をブラウザで開く
where py > nul 2>&1
if %errorlevel%==0 (
  py -3 word_balloon\server.py --open %*
) else (
  python word_balloon\server.py --open %*
)
pause
