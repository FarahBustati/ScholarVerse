"""Small text helpers shared by the catalog tool and the scorer."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be become becoming but by for from have i i'm in into is it like my of
    on or so study studying student the to want wants work working would with degree program
    programs""".split()
)


def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def keywords(*texts: str | None) -> set[str]:
    """Lowercase content words, lightly stemmed, without stopwords."""
    words: set[str] = set()
    for text in texts:
        if not text:
            continue
        for word in re.findall(r"[a-z][a-z+#-]*", text.lower()):
            if word not in STOPWORDS and len(word) > 1:
                words.add(_stem(word))
    return words


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].strip("-") or "student"
