#!/bin/sh
# 気になるワード：サーバーを起動し、リンク一覧（管理者画面・投影画面・スマホ回答）をブラウザで開く
cd "$(dirname "$0")"
exec python3 word_balloon/server.py --open "$@"
