"""The member word lists (M0 S0.1 T5; design 07 §5): `model/lexicon/cs/members/<list>.ttrm`, seeded from the
catalogue and the reference data.

    uv run python scripts/seed_members.py            # write the files
    uv run python scripts/seed_members.py --check    # exit 1 when a committed file differs from the seed

One `def term` per member: its words (`forms`) — the name, a product's nicknames, the declined forms of
`data/reference/declension.csv`, a chain's sub-brands, a DC's town — and, in `description`, the member it belongs to
(its name and code). The catalogue's SKU nicknames are pack words shared by many SKUs, so they are a list of their own
(`pack`, on `sku.pack_format`), one term per word. A word names one member: the editor's lint
(`ttr/lexicon-duplicate-form`, `ttr/lexicon-form-collides-with-name`) holds the lists to that, which is why a factory's
town — an okres's name too — is not a site's word. Every word passes the NH-D8 guard (`checks.czech`).

**The target is a placeholder.** A term names its member through `for:`, and the toolchain the model is pinned to
(ttr-core grammar/v0.21.1) reaches a member only as `<er attribute>.<code>` with `valueLabels` on the attribute: there
is no er layer before M1, and an EAN-13 SKU code cannot be a reference segment at all. So at M0 every term points at
its member vocabulary's column (`db.hartland.<table>.name`; the pack words `sku.pack_format`), which resolves, and
M3 S3.1 T5 re-targets the terms — extending this script — when it completes the lists. Nothing compiles them before
then (`build-lexicon` lands at M3).

The vocabulary itself — the column's values, matched per MV — is declared on the db columns (model/db): names
`searchable method: TYPOS(1)` (or `TOKENS` for the multi-word SKU, product and site names), codes `method: EXACT`.

The carriers' words are their names, and the name without its legal form (`Autodoprava Mašek s.r.o.` is said
`Autodoprava Mašek`), from `data/reference/carrier.csv`.
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from hartland_sim.checks.czech import assert_czech_all
from hartland_sim.io.reference import read
from hartland_sim.setup.classifier.chains import CHAINS

REPO = Path(__file__).resolve().parent.parent
MEMBERS_DIR = REPO / "model/lexicon/cs/members"

# The sub-brands a chain's stores and shops trade under (the classifier's patterns, as they are written). Hand-kept;
# the unit tests hold each to its chain's patterns. COOP's `Tempo` is a pattern but not a word here: `tempo` is a
# common noun ("tempo růstu tržeb"), and a compiled list would bind such a question to COOP.
CHAIN_SUB_BRANDS: dict[str, tuple[str, ...]] = {
    "COOP": ("Jednota", "Konzum", "Terno", "TUTY"),
    "PENNY": ("Penny Market",),
    "OMV": ("Viva",),
    "SHELL": ("Shell Select",),
    "ORLEN": ("Benzina", "Stop Cafe"),
    "MOL": ("Fresh Corner", "Slovnaft"),
}


@dataclass(frozen=True)
class Member:
    code: str  # the member's code, named in the description
    name: str
    forms: tuple[str, ...]
    key: str = ""  # the term id's member part, when not the code


@dataclass(frozen=True)
class WordList:
    name: str  # the file stem and the term-id prefix
    table: str  # the member vocabulary's table
    sources: str  # where the words come from, for the header
    members: tuple[Member, ...]
    column: str = "name"  # the placeholder target's column


def pack_words(catalogue: list[dict[str, object]]) -> tuple[Member, ...]:
    """The catalogue's `nicknames_cs` are pack words — `velká sklenice` is every 200 g jar, of any product — so one
    term per word, not per SKU (a word names one object). Its description says the packs it means."""
    labels = {"jar": "sklenice", "refill": "doplňovací balení"}
    packs: dict[str, set[tuple[str, int]]] = {}
    for row in catalogue:
        for word in _nicknames(row["nicknames_cs"]):  # type: ignore[arg-type]
            packs.setdefault(word, set()).add((str(row["pack_format"]), int(row["net_weight_g"])))  # type: ignore[call-overload]
    members = []
    for word, kinds in packs.items():
        fmt = {f for f, _ in kinds}
        assert len(fmt) == 1, f"pack word {word!r} means more than one pack format: {sorted(fmt)}"
        (pack,) = fmt
        grams = " / ".join(f"{g} g" for g in sorted({g for _, g in kinds}))
        name = f"{labels.get(pack, pack)} {grams}"
        members.append(Member(f"pack_format {pack}", name, (word,), key=re.sub(r"[^a-z]+", "_", _ascii(word))))
    return tuple(members)


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def _forms(*words: str | None) -> tuple[str, ...]:
    """The non-empty words, in order, each once."""
    out: list[str] = []
    for word in words:
        if word and word not in out:
            out.append(word)
    return tuple(out)


def _nicknames(cell: str | None) -> tuple[str, ...]:
    return tuple(n for n in (cell or "").split(";") if n)


def _without_legal_form(name: str) -> str:
    """`Autodoprava Mašek s.r.o.` → `Autodoprava Mašek`: the name as a person says it."""
    return re.sub(r",?\s+(s\.\s?r\.\s?o\.|a\.\s?s\.)$", "", name)


def word_lists() -> list[WordList]:
    """Every list, from the committed reference files."""
    brands = {row["code"]: row["name"] for row in read("brand")}
    catalogue = read("catalogue")
    declension = {row["code"]: row for row in read("declension")}

    def declined(code: str) -> tuple[str, ...]:
        row = declension[code]
        return _forms(row["nominative"], row["genitive"], row["locative"])

    return [
        WordList(
            "sku",
            "sku",
            "data/reference/catalogue.csv (name_cs)",
            tuple(Member(row["sku_code"], row["name_cs"], _forms(row["name_cs"])) for row in catalogue),
        ),
        WordList(
            "pack",
            "sku",
            "data/reference/catalogue.csv (nicknames_cs: the pack words)",
            pack_words(catalogue),
            column="pack_format",
        ),
        WordList(
            "product",
            "product",
            "data/reference/product.csv (name_cs, nicknames_cs)",
            tuple(
                Member(row["code"], row["name_cs"], _forms(row["name_cs"], *_nicknames(row["nicknames_cs"])))
                for row in read("product")
            ),
        ),
        WordList(
            "product_line",
            "product_line",
            "data/reference/product_line.csv (name_cs)",
            tuple(
                Member(row["code"], f"{brands[row['brand']]} {row['name_cs']}", _forms(row["name_cs"]))
                for row in read("product_line")
            ),
        ),
        WordList(
            "brand",
            "brand",
            "data/reference/brand.csv (name)",
            tuple(Member(code, name, _forms(name)) for code, name in brands.items()),
        ),
        WordList(
            "chain",
            "chain",
            "the store classifier's chains (hartland_sim.setup.classifier.chains) and their sub-brands",
            tuple(
                Member(chain.code, chain.name, _forms(chain.name, *CHAIN_SUB_BRANDS.get(chain.code, ())))
                for chain in CHAINS
            ),
        ),
        WordList(
            "kraj",
            "kraj",
            "data/reference/kraj.csv + declension.csv (nominative, genitive, locative)",
            tuple(Member(row["code"], row["name"], declined(row["code"])) for row in read("kraj")),
        ),
        WordList(
            "okres",
            "okres",
            "data/reference/okres.csv + declension.csv (nominative, genitive, locative)",
            tuple(Member(row["code"], row["name"], declined(row["code"])) for row in read("okres")),
        ),
        WordList(
            "site",
            "site",
            "data/reference/site.csv (name; a DC's town)",
            tuple(
                Member(row["code"], row["name"], _forms(row["name"], row["town"] if row["kind"] == "dc" else None))
                for row in read("site")
            ),
        ),
        WordList(
            "carrier",
            "carrier",
            "data/reference/carrier.csv (name; the name without its legal form)",
            tuple(
                Member(row["code"], row["name"], _forms(row["name"], _without_legal_form(row["name"])))
                for row in read("carrier")
            ),
        ),
    ]


def term_id(word_list: str, code: str) -> str:
    return f"{word_list}_{re.sub(r'[^a-z0-9]+', '_', code.lower()).strip('_')}_cs"


def _string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _forms_lines(forms: Iterable[str], indent: str = "    ", width: int = 120) -> list[str]:
    head = f"{indent}forms: ["
    items = [_string(f) for f in forms]
    one = head + ", ".join(items) + "]"
    if len(one) <= width:
        return [one]
    lines = [head]
    line = indent * 2
    for i, item in enumerate(items):
        piece = item + ("," if i < len(items) - 1 else "")
        if len(line) + len(piece) + 1 > width and line.strip():
            lines.append(line.rstrip())
            line = indent * 2
        line += piece + " "
    lines.append(line.rstrip())
    lines.append(f"{indent}]")
    return lines


def render(word_list: WordList) -> str:
    """One list's .ttrm text."""
    header = (
        f"The {word_list.name} word list — seeded by scripts/seed_members.py from {word_list.sources} (M0 S0.1 T5). "
        "Do not edit: edit the sources and re-seed. One term per member, its words in `forms`; `for:` is the member "
        "vocabulary's column until M3 S3.1 T5 re-targets each term to its member (README.md)."
    )
    lines = [
        "package hartland",
        "",
        "model lexicon locale cs",
        "",
        *textwrap.wrap(header, 116, initial_indent="// ", subsequent_indent="// "),
    ]
    for member in word_list.members:
        lines += [
            "",
            f"def term {term_id(word_list.name, member.key or member.code)} {{",
            f"    description: {_string(f'{member.name} ({member.code})')}",
            f"    for: db.hartland.{word_list.table}.{word_list.column}",
            *_forms_lines(member.forms),
            "}",
        ]
    return "\n".join(lines) + "\n"


def rendered() -> dict[Path, str]:
    """Every list's path and text; fails the NH-D8 guard on a word that does not read as Czech."""
    out = {}
    for word_list in word_lists():
        assert_czech_all(
            [form for member in word_list.members for form in member.forms], what=f"members/{word_list.name}"
        )
        out[MEMBERS_DIR / f"{word_list.name}.ttrm"] = render(word_list)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="seed_members", description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="fail when a committed list differs from the seed")
    args = parser.parse_args(argv)
    files = rendered()
    stale = [p for p, text in files.items() if not p.exists() or p.read_text(encoding="utf-8") != text]
    extra = sorted(set(MEMBERS_DIR.glob("*.ttrm")) - set(files))
    if args.check:
        for path in stale:
            print(f"stale: {path.relative_to(REPO)} — run scripts/seed_members.py")
        for path in extra:
            print(f"not seeded: {path.relative_to(REPO)}")
        return 1 if stale or extra else 0
    MEMBERS_DIR.mkdir(parents=True, exist_ok=True)
    for path in stale:
        path.write_text(files[path], encoding="utf-8")
        print(f"wrote {path.relative_to(REPO)}")
    for path in extra:
        print(f"not seeded: {path.relative_to(REPO)} — remove it")
    print(f"seed_members: {len(files)} lists, {len(stale)} written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
