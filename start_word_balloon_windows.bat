@echo off
chcp 65001 > nul
cd /d %~dp0
rem 気になるワード：サーバーを起動し、回答ページをインターネットに公開（同じWi-Fiでなくても回答できる）
rem リンク一覧（管理者画面・投影画面・スマホ回答）をブラウザで開く

if not exist cloudflared.exe (
  echo 初回のみ：インターネット公開用ツール cloudflared をダウンロードしています...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile 'cloudflared.exe'"
  if not exist cloudflared.exe (
    echo cloudflared をダウンロードできませんでした。同じWi-Fi向けで起動します。
    echo 手動で入れる場合は「はじめにお読みください.txt」を見てください。
  )
)

where py > nul 2>&1
if %errorlevel%==0 (
  py -3 word_balloon\server.py --open --tunnel %*
  goto end
)
where python > nul 2>&1
if %errorlevel%==0 (
  python word_balloon\server.py --open --tunnel %*
  goto end
)
echo Python が見つかりません。https://www.python.org/downloads/ からインストールしてください。
echo （インストール時に「Add python.exe to PATH」にチェックを入れてください）
:end
pause
