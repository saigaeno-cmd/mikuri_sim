"""Cloudflare Quick Tunnel で回答ページをインターネットに公開する。

cloudflared（Cloudflare 公式の無料ツール。アカウント不要）を起動し、
表示される https://xxxx.trycloudflare.com を受け取る。URL は起動のたびに変わる。
"""
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
EXE = "cloudflared.exe" if sys.platform == "win32" else "cloudflared"


def find_cloudflared(explicit: str | None, search_dirs) -> str | None:
    """--cloudflared の指定 → 起動フォルダ等に置いた cloudflared → PATH の順に探す。"""
    if explicit:
        return explicit if Path(explicit).exists() else None
    for d in search_dirs:
        p = Path(d) / EXE
        if p.exists():
            return str(p)
    return shutil.which("cloudflared")


class Tunnel:
    def __init__(self, exe: str, port: int):
        self.url: str | None = None
        self.log: list[str] = []
        self._found = threading.Event()
        # Windows では同じコンソールで動かす（黒い画面を閉じると cloudflared も一緒に止まり、公開が残らない）
        self.proc = subprocess.Popen(
            [exe, "tunnel", "--no-autoupdate", "--url", f"http://localhost:{port}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace",
        )
        # 出力を読み続ける（読まないとパイプが詰まって cloudflared が止まる）
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self.log = (self.log + [line.rstrip()])[-30:]
            m = URL_RE.search(line)
            if m and not self.url:
                self.url = m.group(0) + "/"
                self._found.set()
        self._found.set()   # 終了した（失敗）

    def wait_url(self, timeout: float = 40) -> str | None:
        self._found.wait(timeout)
        return self.url

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
