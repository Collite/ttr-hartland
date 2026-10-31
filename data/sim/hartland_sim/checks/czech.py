"""The NH-D8 guard: no English in any generated data or document.

`assert_czech(text)` fails on (1) text that is not NFC, (2) any English stop-word, and (3) a text of 200 characters or
more without a single Czech diacritic — long Czech prose always has one, so its absence means English, or Czech
typed without accents. `assert_czech_all(texts)` checks a column or a set of labels: each text alone, then the
diacritic rule over the texts together, so a column of short English labels fails as a whole. Every later renderer
test runs its output through them (DOC.1–DOC.3, the labels of G0 S0.2).

The rules:
- Generated text is NFC: Postgres equality, the lexicon's members and the glossary compare code points, so NFD is
  an error of the source — normalise there, not here.
- The stop-word list leaves out the words Czech and Slovak share with English (`a`, `i`, `to`, `do`, `on`, `no`,
  `by`, `my`, `so`, `ten`, `had`, `her`, `most`); Slovak passes (⚑NH-P-8).
- A hyphenated compound (`roll-up`, `know-how`, `check-in`) is one token and never a stop-word: Czech writes its
  loanwords that way.
- `allow` names the slot values a renderer filled (store, customer and brand names), removed before the checks —
  pass the names, never template text. Capitalised words are not skipped: sentence-initial English would pass.
- Acronyms are an explicit set (`ACRONYMS`); any other all-capitals token is a word like the rest.

The limit: one short English label among many Czech ones can still pass. The guard is a smoke detector, not a
classifier. Apply it to human-readable text only — a letter's body, a label, a note — not to protocol syntax
(RFC 5322 header names, JSON keys, codes).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

ENGLISH_STOP_WORDS: frozenset[str] = frozenset(
    """
    the and of in is it for with that this these those are was were be been being have has having
    from or but not you your yours we our ours they their them he him his she its an as at if into
    which what when where who whom why how there here all any each both few some such than then very
    just also only other again about after before between through during without under over up out
    will would should could can cannot does did done please thank thanks dear regards yes now
    """.split()
)

# Add an acronym here when a real one collides with a stop-word (exact case).
ACRONYMS: frozenset[str] = frozenset({"IT"})

CZECH_DIACRITICS: frozenset[str] = frozenset("áčďéěíňóřšťúůýž" + "äĺľôŕ")

LONG_TEXT = 200

_WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")


def _without(text: str, allow: Iterable[str]) -> str:
    for name in sorted({name for name in allow if name}, key=len, reverse=True):
        text = text.replace(name, " ")
    return text


def english_words(text: str, allow: Iterable[str] = ()) -> list[str]:
    """The English stop-words in NFC `text` (minus the `allow` names), lower-cased, in order of appearance."""
    return [
        token.lower()
        for token in _WORD.findall(_without(text, allow))
        if "-" not in token and token not in ACRONYMS and token.lower() in ENGLISH_STOP_WORDS
    ]


def has_diacritics(text: str) -> bool:
    """Whether NFC `text` carries at least one Czech (or Slovak) diacritic."""
    return any(ch in CZECH_DIACRITICS for ch in text.lower())


def assert_czech(text: str, *, what: str = "text", allow: Iterable[str] = ()) -> None:
    """Fail unless `text` reads as Czech; `what` names it in the message (a file, a column, a template slot), `allow`
    the slot values filled into it (names, brands)."""
    preview = text[:80].replace("\n", " ")
    if not unicodedata.is_normalized("NFC", text):
        raise AssertionError(f"{what}: not NFC (normalise the source, not the check) in {preview!r}…")
    checked = _without(text, allow)
    found = english_words(checked)
    if found:
        raise AssertionError(f"{what}: English words {sorted(set(found))} in {preview!r}…")
    if len(checked) >= LONG_TEXT and not has_diacritics(checked):
        raise AssertionError(f"{what}: {len(checked)} characters without a Czech diacritic in {preview!r}…")


def assert_czech_all(texts: Iterable[str], *, what: str, allow: Iterable[str] = ()) -> None:
    """`assert_czech` on each text (`what[i]`), then the diacritic rule over the texts joined — for a column or a
    set of labels, where each label alone is too short to tell."""
    texts = list(texts)
    allow = tuple(allow)
    for i, text in enumerate(texts):
        assert_czech(text, what=f"{what}[{i}]", allow=allow)
    joined = " ".join(_without(text, allow) for text in texts)
    if len(joined) >= LONG_TEXT and not has_diacritics(joined):
        raise AssertionError(f"{what}: {len(texts)} texts, {len(joined)} characters without a Czech diacritic")
