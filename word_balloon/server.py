"""会場アンケート「気になるワード」バルーン表示サーバー（標準ライブラリのみ）。

    python word_balloon/server.py [--port 8000] [--tunnel] [--open]

  --tunnel を付けると Cloudflare Quick Tunnel で回答ページをインターネットに公開し、
  同じ Wi-Fi でなくても（モバイル回線からでも）回答できるようにする。

  /         参加者の回答ページ（スマホ）
  /screen   会場スクリーン用の表示ページ
  /admin    管理ページ（要確認ワードの許可・非表示、削除、リセット。起動時に表示される管理キーが必要）
"""
import argparse
import html
import json
import os
import secrets
import signal
import socket
import sys
import threading
import time
import webbrowser
from collections import defaultdict, deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from word_balloon.store import APPROVED, REJECTED, WordStore  # noqa: E402
from word_balloon.tunnel import Tunnel, find_cloudflared  # noqa: E402

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
PAGES = {"/": "answer.html", "/screen": "screen.html", "/admin": "admin.html"}
MAX_BODY = 4096
# 連続投稿の制限（1分あたり）。会場 Wi-Fi では大勢が同じ IP になるので、IP の上限は大きめ
LIMIT_PER_CLIENT = 10
LIMIT_PER_IP = 120


class RateLimiter:
    def __init__(self, limit: int, window: float = 60.0):
        self.limit, self.window = limit, window
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            if len(self._hits) > 20000:   # 古いキーを掃除
                for k in [k for k, v in self._hits.items() if not v]:
                    del self._hits[k]
            return True


def lan_ip() -> str:
    """同じ Wi-Fi の参加者がアクセスするための IP（取れなければ localhost）。"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def make_links(answer_url: str, port: int, admin_key: str, public: bool = False) -> dict:
    """PC 用（管理者画面・投影画面）は localhost、スマホ用（回答ページ）は LAN/公開 URL。
    管理者画面のリンクには管理キーを # 以降に付ける（# 以降はサーバーに送られず、ページが読んで消す）。"""
    local = f"http://localhost:{port}"
    return {
        "admin": f"{local}/admin#key={quote(admin_key)}",
        "screen": f"{local}/screen",
        "answer": answer_url,
        "answer_note": ("インターネットに公開中です。Wi‑Fi でもモバイル回線でも、どのスマホからでも開けます。"
                        "URL は起動のたびに変わります。" if public else
                        "このPCと同じWi‑Fiにつないだスマホで開きます。"),
    }


def write_links_page(path: Path, links: dict) -> None:
    """3つのリンクをまとめたページ（管理キーを含むので data/ に置き、配布しない）。"""
    tpl = (STATIC / "links.html").read_text(encoding="utf-8")
    for k, v in links.items():
        tpl = tpl.replace("{{" + k + "}}", html.escape(v, quote=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tpl, encoding="utf-8")


def read_list(path) -> list[str]:
    """1行1語のリスト（# で始まる行と空行は無視）。"""
    p = Path(path)
    if not p.exists():
        return []
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.lstrip().startswith("#")]


def make_handler(store: WordStore, admin_key: str, info: dict):
    """info["answer_url"] は参加者に案内する URL（トンネル開通後に書き換わる）。"""
    by_client, by_ip = RateLimiter(LIMIT_PER_CLIENT), RateLimiter(LIMIT_PER_IP)

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

        def _client_ip(self) -> str:
            ip = self.client_address[0]
            if ip in ("127.0.0.1", "::1"):   # トンネル経由は cloudflared（localhost）から来る
                ip = self.headers.get("CF-Connecting-IP", ip)
            return ip

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
                return self._json({"answer_url": info["answer_url"]})
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            path = urlparse(self.path).path
            data = self._body()
            if path == "/api/answer":
                client = data.get("client")
                client = client[:64] if isinstance(client, str) and client else "-"
                ip = self._client_ip()
                if not by_ip.allow(ip) or not by_client.allow(f"{ip}|{client}"):
                    return self._json({"ok": False, "error": "送信が多すぎます。少し待ってからもう一度お試しください"},
                                      HTTPStatus.TOO_MANY_REQUESTS)
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


def _stop(*_):
    raise KeyboardInterrupt


def start_tunnel(explicit, port):
    """トンネルを開き URL が出るまで待つ。失敗したら None（同じ Wi-Fi 向けの URL のまま続ける）。"""
    root = HERE.parent
    exe = find_cloudflared(explicit, [root, HERE, Path.cwd()])
    if not exe:
        print("！ cloudflared が見つからないため、インターネット公開はせず同じ Wi-Fi 向けで起動します。")
        print(f"   {root} に cloudflared を置いてください（README 参照）。")
        return None
    print("インターネット公開用のトンネルを準備しています（最大40秒）…")
    try:
        t = Tunnel(exe, port)
    except OSError as e:
        print(f"！ cloudflared を起動できませんでした: {e}")
        return None
    if not t.wait_url(40):
        print("！ トンネルの URL を取得できませんでした。インターネット接続を確認してください。")
        print("   同じ Wi-Fi 向けの URL で起動します。cloudflared の出力:")
        for line in t.log[-8:]:
            print("     " + line)
        t.stop()
        return None
    return t


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
    p.add_argument("--open", action="store_true", help="起動後にリンク一覧をブラウザで開く")
    p.add_argument("--tunnel", action="store_true",
                   help="Cloudflare Quick Tunnel で回答ページをインターネットに公開する（cloudflared が必要）")
    p.add_argument("--cloudflared", help="cloudflared の場所（省略時は起動フォルダ・PATH から探す）")
    a = p.parse_args(argv)

    store = WordStore(Path(a.data), ng_words=read_list(a.ng), review_words=read_list(a.review),
                      review_all=a.review_all)
    admin_key = os.environ.get("WORD_BALLOON_ADMIN_KEY") or secrets.token_urlsafe(12)
    info = {"answer_url": a.public_url or f"http://{lan_ip()}:{a.port}/"}

    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(store, admin_key, info))
    tunnel = None
    if a.tunnel:
        tunnel = start_tunnel(a.cloudflared, a.port)
        if tunnel and tunnel.url:
            info["answer_url"] = tunnel.url
    links = make_links(info["answer_url"], a.port, admin_key, public=bool(tunnel or a.public_url))
    links_file = Path(a.data).parent / "links.html"
    write_links_page(links_file, links)
    print("ワードバルーン サーバーを起動しました（Ctrl+C で終了）")
    print(f"  管理者画面（PC）    : {links['admin']}")
    print(f"  投影画面（PC）      : {links['screen']}")
    print(f"  回答ページ（スマホ）: {links['answer']}")
    print(f"  リンク一覧          : {links_file.resolve().as_uri()}")
    if tunnel and tunnel.url:
        print("  ※ 回答ページはインターネットに公開中です（同じ Wi-Fi でなくても回答できます）。")
        print("     このウィンドウを閉じると公開も止まります。URL は起動のたびに変わります。")
    if a.review_all:
        print("  ※ すべての新しいワードは管理ページで許可するまでスクリーンに表示されません")
    if a.open:
        webbrowser.open(links_file.resolve().as_uri())
    for name in ("SIGTERM", "SIGHUP", "SIGBREAK"):   # 終了の合図でもトンネルを確実に止める
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _stop)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        if tunnel:
            tunnel.stop()


if __name__ == "__main__":
    main()
