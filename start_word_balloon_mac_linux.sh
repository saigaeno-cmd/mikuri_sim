#!/bin/sh
# 気になるワード：サーバーを起動し、リンク一覧（管理者画面・投影画面・スマホ回答）をブラウザで開く
# cloudflared が入っていれば（brew install cloudflared）回答ページをインターネットに公開する
cd "$(dirname "$0")"
exec python3 word_balloon/server.py --open --tunnel "$@"
