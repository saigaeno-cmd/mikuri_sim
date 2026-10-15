"""回答ワードの集計（スレッドセーフ・JSON 保存）。

同じワードは表記ゆれ（全角/半角・大文字/小文字・前後の空白）をそろえて1つに数える。
画面には最初に回答された表記を出す。
"""
import json
import threading
import time
import unicodedata
from pathlib import Path

MAX_LEN = 20


def normalize(word: str) -> str:
    """集計用のキー。NFKC で全角英数・半角カナをそろえ、小文字化し、空白を詰める。"""
    s = unicodedata.normalize("NFKC", word or "")
    s = "".join(s.split())
    return s.casefold()


def clean(word: str) -> str:
    """画面に出す表記。NFKC 後に前後の空白を除き、連続空白を1つにする。"""
    s = unicodedata.normalize("NFKC", word or "")
    return " ".join(s.split())


class WordStore:
    def __init__(self, path: Path | None = None, ng_words=()):
        self.path = Path(path) if path else None
        self.ng = {normalize(w) for w in ng_words if normalize(w)}
        self._lock = threading.Lock()
        self.words: dict[str, dict] = {}   # key -> {"word", "count", "first", "last"}
        self.version = 0
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.words = data.get("words", {})
            self.version = data.get("version", 0)

    def add(self, word: str) -> tuple[bool, str]:
        """回答を1件追加する。戻り値は (受付可否, 表示表記 or エラー理由)。"""
        text = clean(word)
        key = normalize(text)
        if not key:
            return False, "ワードを入力してください"
        if len(text) > MAX_LEN:
            return False, f"{MAX_LEN}文字以内で入力してください"
        if any(ng in key for ng in self.ng):
            return False, "そのワードは受け付けられません"
        now = time.time()
        with self._lock:
            item = self.words.get(key)
            if item is None:
                item = self.words[key] = {"word": text, "count": 0, "first": now}
            item["count"] += 1
            item["last"] = now
            self.version += 1
            self._save()
            return True, item["word"]

    def delete(self, key: str) -> bool:
        with self._lock:
            if self.words.pop(normalize(key), None) is None:
                return False
            self.version += 1
            self._save()
            return True

    def reset(self) -> None:
        with self._lock:
            self.words = {}
            self.version += 1
            self._save()

    def snapshot(self) -> dict:
        with self._lock:
            items = sorted(
                ({"key": k, **v} for k, v in self.words.items()),
                key=lambda x: (-x["count"], x["first"]),
            )
            return {
                "version": self.version,
                "total": sum(x["count"] for x in items),
                "words": items,
            }

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"version": self.version, "words": self.words}, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self.path)
