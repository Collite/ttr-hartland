"""The member word lists (scripts/seed_members.py; M0 S0.1 T5, design 07 §5): the hand-kept declension table, the
committed lists against the seed, the NH-D8 guard over every word, and one word for one member."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from hartland_sim.io.reference import read
from hartland_sim.setup.classifier import chains

REPO = Path(__file__).resolve().parents[3]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("seed_members", REPO / "scripts/seed_members.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["seed_members"] = module
    spec.loader.exec_module(module)
    return module


seed = _load()


def test_the_declension_table_covers_every_kraj_and_okres_under_its_name() -> None:
    declension = {row["code"]: row for row in read("declension")}
    names = {row["code"]: row["name"] for row in read("kraj")} | {row["code"]: row["name"] for row in read("okres")}
    assert set(declension) == set(names)
    for code, name in names.items():
        assert declension[code]["nominative"] == name, code


def test_a_few_declined_forms() -> None:
    declension = {row["code"]: row for row in read("declension")}
    assert (declension["CZ064"]["genitive"], declension["CZ064"]["locative"]) == (
        "Jihomoravského kraje",
        "Jihomoravském kraji",
    )
    assert (declension["CZ0642"]["genitive"], declension["CZ0642"]["locative"]) == ("Brna-města", "Brně-městě")
    assert (declension["CZ0100-05"]["genitive"], declension["CZ0100-05"]["locative"]) == ("Prahy 5", "Praze 5")
    assert declension["CZ0427"]["locative"] == "Ústí nad Labem"  # Ústí does not decline


def test_the_committed_lists_are_the_seed() -> None:
    assert seed.main(["--check"]) == 0


def test_every_list_and_its_size() -> None:
    sizes = {word_list.name: len(word_list.members) for word_list in seed.word_lists()}
    assert sizes == {
        "sku": 123,
        "pack": 3,
        "product": 48,
        "product_line": 11,
        "brand": 2,
        "chain": 22,
        "kraj": 14,
        "okres": 99,
        "site": 18,
        "carrier": 8,
    }


def test_a_word_names_one_member() -> None:
    owner: dict[str, str] = {}
    for word_list in seed.word_lists():
        for member in word_list.members:
            for form in member.forms:
                here = f"{word_list.name}:{member.code}"
                assert owner.setdefault(form, here) == here, f"{form!r}: {owner[form]} and {here}"


def test_the_nicknames_reach_their_product_and_the_pack_words_their_pack() -> None:
    by_name = {word_list.name: word_list for word_list in seed.word_lists()}
    lists = {name: {m.code: m for m in word_list.members} for name, word_list in by_name.items()}
    assert {"zlatý kafe", "zlatá"} <= set(lists["product"]["GOLD"].forms)
    assert "klasik" in lists["product"]["CLASSIC"].forms
    assert "trojka" in lists["product"]["3V1"].forms
    packs = {m.forms[0]: m.name for m in by_name["pack"].members}
    assert packs == {
        "malá sklenice": "sklenice 50 g",
        "velká sklenice": "sklenice 200 g",
        "doplňovačka": "doplňovací balení 150 g / 300 g",
    }
    assert lists["site"]["DC-JIRNY"].forms == ("Distribuční centrum Jirny", "Jirny")
    assert lists["site"]["FA-KOLIN"].forms == ("Závod Kolín",)  # Kolín is the okres's word


def test_a_carrier_is_its_name_and_its_name_without_the_legal_form() -> None:
    carriers = {m.code: m for m in next(w for w in seed.word_lists() if w.name == "carrier").members}
    assert carriers["DOP-J1"].forms == ("Autodoprava Mašek s.r.o.", "Autodoprava Mašek")
    assert carriers["DOP-J3"].forms == ("Doprava Horák a syn s.r.o.", "Doprava Horák a syn")


def test_every_chain_word_is_its_chain_to_the_classifier() -> None:
    """A sub-brand is a word of its chain because the classifier reads it as that chain: a renamed pattern is seen."""
    for code, words in seed.CHAIN_SUB_BRANDS.items():
        for word in words:
            chain = chains.match(brand=word)
            assert chain is not None and chain.code == code, f"{word!r}: {chain and chain.code} (want {code})"


def test_tempo_is_a_pattern_but_not_a_word() -> None:
    """COOP Tempo is a COOP format (the classifier keeps it), but `tempo` is a common noun (review-202 ⚑R202-A)."""
    assert chains.match(name="COOP Tempo").code == "COOP"  # type: ignore[union-attr]
    words = {form.casefold() for w in seed.word_lists() for m in w.members for form in m.forms}
    assert "tempo" not in words


def test_the_write_mode_names_a_list_it_did_not_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    members = tmp_path / "model/lexicon/cs/members"
    monkeypatch.setattr(seed, "REPO", tmp_path)
    monkeypatch.setattr(seed, "MEMBERS_DIR", members)
    members.mkdir(parents=True)
    (members / "old.ttrm").write_text("package hartland\n", encoding="utf-8")
    assert seed.main([]) == 0
    assert "not seeded: model/lexicon/cs/members/old.ttrm — remove it" in capsys.readouterr().out


def test_every_word_reads_as_czech() -> None:
    seed.rendered()  # assert_czech_all per list, or AssertionError naming the list and the word


def test_a_term_id_is_an_identifier() -> None:
    assert seed.term_id("sku", "2931000000012") == "sku_2931000000012_cs"
    assert seed.term_id("okres", "CZ0100-01") == "okres_cz0100_01_cs"
