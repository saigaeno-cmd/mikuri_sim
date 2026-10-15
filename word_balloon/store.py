"""回答ワードの集計（スレッドセーフ・JSON 保存）。

同じワードは表記ゆれ（全角/半角・大文字/小文字・前後の空白）をそろえて1つに数える。
画面には最初に回答された表記を出す。

ワードの状態（status）
  approved  スクリーンに表示する
  pending   承認待ち。要確認ワードを含む回答（review_all なら全回答）は管理者が許可するまで表示しない
  rejected  管理者が非表示にしたワード。同じワードがまた回答されても表示しない
"""
import json
import threading
import time
import unicodedata
from pathlib import Path

MAX_LEN = 20
APPROVED, PENDING, REJECTED = "approved", "pending", "rejected"
STATUSES = (APPROVED, PENDING, REJECTED)


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
    def __init__(self, path: Path | None = None, ng_words=(), review_words=(), review_all=False):
        self.path = Path(path) if path else None
        self.ng = {normalize(w) for w in ng_words if normalize(w)}
        self.review = {normalize(w) for w in review_words if normalize(w)}
        self.review_all = review_all
        self._lock = threading.Lock()
        self.words: dict[str, dict] = {}   # key -> {"word", "count", "first", "last"}
        self.version = 0
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.words = data.get("words", {})
            self.version = data.get("version", 0)
            for item in self.words.values():
                item.setdefault("status", APPROVED)

    def needs_review(self, key: str) -> bool:
        return self.review_all or any(w in key for w in self.review)

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
                status = PENDING if self.needs_review(key) else APPROVED
                item = self.words[key] = {"word": text, "count": 0, "first": now, "status": status}
            item["count"] += 1
            item["last"] = now
            self.version += 1
            self._save()
            return True, item["word"]

    def set_status(self, key: str, status: str) -> bool:
        if status not in STATUSES:
            return False
        with self._lock:
            item = self.words.get(normalize(key))
            if item is None:
                return False
            item["status"] = status
            self.version += 1
            self._save()
            return True

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

    def snapshot(self, include_hidden: bool = False) -> dict:
        """集計結果。スクリーン用（既定）は表示を許可したワードだけ、管理用は全ワード。"""
        with self._lock:
            items = sorted(
                ({"key": k, **v} for k, v in self.words.items()
                 if include_hidden or v["status"] == APPROVED),
                key=lambda x: (-x["count"], x["first"]),
            )
            return {
                "version": self.version,
                "total": sum(x["count"] for x in items),
                "pending": sum(1 for v in self.words.values() if v["status"] == PENDING),
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
