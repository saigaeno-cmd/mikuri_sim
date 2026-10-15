"""会場アンケート「気になるワード」バルーン表示サーバー（標準ライブラリのみ）。

    python word_balloon/server.py [--port 8000] [--data word_balloon/data/answers.json]

  /         参加者の回答ページ（スマホ）
  /screen   会場スクリーン用の表示ページ
  /admin    管理ページ（要確認ワードの許可・非表示、削除、リセット。起動時に表示される管理キーが必要）
"""
import argparse
import json
import os
import secrets
import socket
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from word_balloon.store import APPROVED, REJECTED, WordStore  # noqa: E402

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
PAGES = {"/": "answer.html", "/screen": "screen.html", "/admin": "admin.html"}
MAX_BODY = 4096


def lan_ip() -> str:
    """同じ Wi-Fi の参加者がアクセスするための IP（取れなければ localhost）。"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def read_list(path) -> list[str]:
    """1行1語のリスト（# で始まる行と空行は無視）。"""
    p = Path(path)
    if not p.exists():
        return []
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.lstrip().startswith("#")]


def make_handler(store: WordStore, admin_key: str, answer_url: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "WordBalloon/1.0"

        def log_message(self, fmt, *args):  # アクセスログは出さない
            pass

        def _send(self, status, body: bytes, ctype: str):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status=HTTPStatus.OK):
            self._send(status, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                return {}
            try:
                data = json.loads(self.rfile.read(n).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return {}
            return data if isinstance(data, dict) else {}

        def do_GET(self):
            path = urlparse(self.path).path
            if path in PAGES:
                body = (STATIC / PAGES[path]).read_bytes()
                return self._send(HTTPStatus.OK, body, "text/html; charset=utf-8")
            if path == "/api/words":
                return self._json(store.snapshot())
            if path == "/api/info":
                return self._json({"answer_url": answer_url})
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            path = urlparse(self.path).path
            data = self._body()
            if path == "/api/answer":
                word = data.get("word")
                ok, msg = store.add(word if isinstance(word, str) else "")
                if ok:
                    return self._json({"ok": True, "word": msg})
                return self._json({"ok": False, "error": msg}, HTTPStatus.BAD_REQUEST)
            if path.startswith("/api/admin/"):
                if not secrets.compare_digest(self.headers.get("X-Admin-Key", ""), admin_key):
                    return self._json({"ok": False, "error": "管理キーが違います"}, HTTPStatus.FORBIDDEN)
                if path == "/api/admin/check":
                    return self._json({"ok": True})
                if path == "/api/admin/words":
                    return self._json({"ok": True, **store.snapshot(include_hidden=True)})
                if path in ("/api/admin/approve", "/api/admin/reject"):
                    key = data.get("key")
                    status = APPROVED if path.endswith("approve") else REJECTED
                    return self._json({"ok": store.set_status(key if isinstance(key, str) else "", status)})
                if path == "/api/admin/delete":
                    key = data.get("key")
                    return self._json({"ok": store.delete(key if isinstance(key, str) else "")})
                if path == "/api/admin/reset":
                    store.reset()
                    return self._json({"ok": True})
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    return Handler


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--data", default=str(HERE / "data" / "answers.json"), help="回答の保存先（再起動しても残る）")
    p.add_argument("--ng", default=str(HERE / "ng_words.txt"), help="受け付けないワード（1行1語、部分一致）")
    p.add_argument("--review", default=str(HERE / "review_words.txt"),
                   help="要確認ワード（1行1語、部分一致）。含む回答は管理者が許可するまでスクリーンに出さない")
    p.add_argument("--review-all", action="store_true", help="すべての新しいワードを管理者の許可制にする")
    p.add_argument("--public-url", help="参加者に案内する URL（省略時は LAN の IP から作成）")
    a = p.parse_args(argv)

    store = WordStore(Path(a.data), ng_words=read_list(a.ng), review_words=read_list(a.review),
                      review_all=a.review_all)
    admin_key = os.environ.get("WORD_BALLOON_ADMIN_KEY") or secrets.token_urlsafe(6)
    answer_url = a.public_url or f"http://{lan_ip()}:{a.port}/"

    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(store, admin_key, answer_url))
    base = answer_url.rstrip("/")
    print("ワードバルーン サーバーを起動しました（Ctrl+C で終了）")
    print(f"  回答ページ（参加者）: {answer_url}")
    print(f"  スクリーン表示     : {base}/screen")
    print(f"  管理ページ         : {base}/admin   管理キー: {admin_key}")
    if a.review_all:
        print("  ※ すべての新しいワードは管理ページで許可するまでスクリーンに表示されません")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
