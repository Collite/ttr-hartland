"""The NH-D8 guard: no English in any generated data or document.

`assert_czech(text)` fails on (1) any English stop-word, and (2) a text of 200 characters or more without a single
Czech diacritic — long Czech prose always has one, so its absence means English, or Czech typed without accents.
Every later renderer test runs its output through it (DOC.1–DOC.3, the labels of G0 S0.2).

Apply it to human-readable text only — a letter's body, a label, a note — not to protocol syntax (RFC 5322 header
names, JSON keys, codes). The stop-word list leaves out the words Czech and Slovak share with English
(`a`, `i`, `to`, `do`, `on`, `no`, `by`, `my`, `so`, `ten`, `had`, `her`, `most`); Slovak passes (⚑NH-P-8). An
all-capitals token is an acronym (`IT`, `OMV`), not a word.
"""

from __future__ import annotations

import re

ENGLISH_STOP_WORDS: frozenset[str] = frozenset(
    """
    the and of in is it for with that this these those are was were be been being have has having
    from or but not you your yours we our ours they their them he him his she its an as at if into
    which what when where who whom why how there here all any each both few some such than then very
    just also only other again about after before between through during without under over up out
    will would should could can cannot does did done please thank thanks dear regards yes now
    """.split()
)

CZECH_DIACRITICS: frozenset[str] = frozenset("áčďéěíňóřšťúůýž" + "äĺľôŕ")

LONG_TEXT = 200

_WORD = re.compile(r"[^\W\d_]+")


def english_words(text: str) -> list[str]:
    """The English stop-words in `text`, lower-cased, in order of appearance (acronyms skipped)."""
    return [
        word.lower()
        for word in _WORD.findall(text)
        if not (len(word) > 1 and word.isupper()) and word.lower() in ENGLISH_STOP_WORDS
    ]


def has_diacritics(text: str) -> bool:
    """Whether `text` carries at least one Czech (or Slovak) diacritic."""
    return any(ch in CZECH_DIACRITICS for ch in text.lower())


def assert_czech(text: str, *, what: str = "text") -> None:
    """Fail unless `text` reads as Czech; `what` names it in the message (a file, a column, a template slot)."""
    preview = text[:80].replace("\n", " ")
    found = english_words(text)
    if found:
        raise AssertionError(f"{what}: English words {sorted(set(found))} in {preview!r}…")
    if len(text) >= LONG_TEXT and not has_diacritics(text):
        raise AssertionError(f"{what}: {len(text)} characters without a Czech diacritic in {preview!r}…")
