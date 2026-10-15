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


def test_review_words_are_hidden_until_approved():
    s = WordStore(review_words=["政治"])
    assert s.add("政治とお金") == (True, "政治とお金")   # 回答者には普通に受け付けたと返す
    s.add("政治とお金"); s.add("自動運転")
    screen = s.snapshot()
    assert [w["word"] for w in screen["words"]] == ["自動運転"]
    assert screen["total"] == 1 and screen["pending"] == 1
    admin = s.snapshot(include_hidden=True)
    assert {w["word"]: w["status"] for w in admin["words"]} == {"政治とお金": "pending", "自動運転": "approved"}
    assert s.set_status("政治とお金", "approved")
    assert [(w["word"], w["count"]) for w in s.snapshot()["words"]] == [("政治とお金", 2), ("自動運転", 1)]
    assert s.set_status("政治とお金", "rejected")
    s.add("政治とお金")   # 非表示にしたワードはまた回答されても出ない
    assert [w["word"] for w in s.snapshot()["words"]] == ["自動運転"]
    assert not s.set_status("政治とお金", "bogus")


def test_review_all_and_legacy_file(tmp_path):
    s = WordStore(review_all=True)
    s.add("なんでも")
    assert s.snapshot()["words"] == []
    path = tmp_path / "old.json"   # status が無い以前の保存ファイルは表示扱い
    path.write_text('{"version": 1, "words": {"a": {"word": "a", "count": 1, "first": 0, "last": 0}}}', encoding="utf-8")
    assert [w["word"] for w in WordStore(path).snapshot()["words"]] == ["a"]


@pytest.fixture
def server():
    store = WordStore(review_words=["NG候補"])
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
    assert call(server + "/api/answer", {"word": "NG候補です"})[1]["ok"]
    assert call(server + "/api/words")[1]["total"] == 1      # スクリーンには出ない
    key = {"X-Admin-Key": "secret"}
    assert call(server + "/api/admin/words", {})[0] == 403
    words = call(server + "/api/admin/words", {}, key)[1]["words"]
    assert {w["word"]: w["status"] for w in words}["NG候補です"] == "pending"
    assert call(server + "/api/admin/approve", {"key": "NG候補です"}, key) == (200, {"ok": True})
    assert call(server + "/api/words")[1]["total"] == 2
    assert call(server + "/api/admin/reject", {"key": "NG候補です"}, key) == (200, {"ok": True})
    assert call(server + "/api/words")[1]["total"] == 1
    assert call(server + "/api/admin/reset", {})[0] == 403
    assert call(server + "/api/admin/reset", {}, {"X-Admin-Key": "secret"}) == (200, {"ok": True})
    assert call(server + "/api/words")[1]["total"] == 0
    for page in ("/", "/screen", "/admin"):
        with urllib.request.urlopen(server + page) as r:
            assert r.status == 200 and b"<html" in r.read()
