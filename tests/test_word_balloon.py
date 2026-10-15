import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from word_balloon.server import make_handler
from word_balloon.store import MAX_LEN, WordStore, normalize


def test_normalize_merges_width_case_and_spaces():
    assert normalize("ＡＩ") == normalize("ai") == normalize(" A I ")
    assert normalize("ｶｰﾎﾞﾝ") == normalize("カーボン")


def test_add_counts_same_word_and_keeps_first_form(tmp_path):
    s = WordStore(tmp_path / "a.json")
    assert s.add("AI") == (True, "AI")
    assert s.add("ａｉ") == (True, "AI")
    s.add("自動運転")
    snap = s.snapshot()
    assert snap["total"] == 3
    assert [(w["word"], w["count"]) for w in snap["words"]] == [("AI", 2), ("自動運転", 1)]
    # 再起動しても残る
    again = WordStore(tmp_path / "a.json").snapshot()
    assert again["total"] == 3 and again["version"] == snap["version"]


def test_rejects_empty_long_and_ng_words():
    s = WordStore(ng_words=["ばか"])
    assert not s.add("   ")[0]
    assert not s.add("あ" * (MAX_LEN + 1))[0]
    assert not s.add("おおばかもの")[0]
    assert s.snapshot()["total"] == 0


def test_delete_and_reset():
    s = WordStore()
    s.add("a"); s.add("b"); s.add("b")
    assert s.delete("Ｂ")
    assert not s.delete("zzz")
    assert [w["word"] for w in s.snapshot()["words"]] == ["a"]
    s.reset()
    assert s.snapshot()["words"] == []


@pytest.fixture
def server():
    store = WordStore()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store, "secret", "http://x/"))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def call(url, body=None, headers=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_api(server):
    assert call(server + "/api/answer", {"word": "自動運転"}) == (200, {"ok": True, "word": "自動運転"})
    assert call(server + "/api/answer", {"word": ""})[0] == 400
    assert call(server + "/api/answer", {"word": 123})[0] == 400
    status, snap = call(server + "/api/words")
    assert status == 200 and snap["total"] == 1
    assert call(server + "/api/admin/reset", {})[0] == 403
    assert call(server + "/api/admin/reset", {}, {"X-Admin-Key": "secret"}) == (200, {"ok": True})
    assert call(server + "/api/words")[1]["total"] == 0
    for page in ("/", "/screen", "/admin"):
        with urllib.request.urlopen(server + page) as r:
            assert r.status == 200 and b"<html" in r.read()
