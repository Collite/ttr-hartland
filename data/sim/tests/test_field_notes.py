"""The visit notes (G2 S2.1 T6, `loop.field.note`): a recorded visit's note is a sentence of its outcome from
`visit_note_templates.yaml` (an acquisition's or a merchandising visit's `no_order` its kind's) and one of each
situation it met, as its rows say it — a mandatory promotion running, executed or not, a line out of stock, a
competitor's activity by its kind, a display lent — 2–4 sentences, their slots filled from the visit (a product on sale
that day by its name or nickname, the competitor, the promotion, an objection), stream `note`; A-6's sightings go in
hand-written and verbatim; a visit to a shop closed or gone met nothing (review-231 R2, R3); an unrecorded visit and
every visit of A-8's rep have none. The visits are the opening `World`'s over two weeks of 2023 (Zimní polévky, a
mandatory promotion, runs), their outcomes as drawn and the order ones set by hand."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace
from datetime import date, timedelta

import pytest

from hartland_sim.checks.czech import assert_czech, has_diacritics
from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.io.reference import read
from hartland_sim.loop import field
from hartland_sim.loop.field import FIRST_REFUSAL, PLANTED_REFUSALS, PLANTED_SIGHTINGS, Crew, Listing, Visit
from hartland_sim.setup.rows import Row
from hartland_sim.setup.run import Setup

WORLD = WorldConfig.load()
STORIES = StoriesConfig.load()
DAYS = [date(2023, 1, 9) + timedelta(days=k) for k in range(12) if (date(2023, 1, 9) + timedelta(days=k)).weekday() < 5]
SLOT = re.compile(r"\{\{ (\w+) \}\}")
NAMES = [
    *(row["name_cs"] for row in read("product")),
    *(nick for row in read("product") for nick in (row["nicknames_cs"] or "").split(";") if nick),
    *(row["name"] for row in read("competitor")),
]


@pytest.fixture(scope="module")
def crew(opening: Setup) -> Crew:
    return Crew.of(opening.world, WORLD, STORIES)


@pytest.fixture(scope="module")
def noted(opening: Setup, crew: Crew) -> list[Visit]:
    world = opening.world
    out = []
    for day in DAYS:
        visits = field.visit(world, crew, day, field.calls(world, crew, day, field.plan(world, day)))
        for v in visits:
            if v.converts:
                v.outcome = "order"
        observations = field.observe(world, crew, day, visits)
        sightings = field.sight(world, crew, day, visits)
        field.note(world, crew, day, visits, observations, sightings)
        out += visits
    return out


def _sentences(note: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", note) if s]


def test_a_recorded_visit_s_note_is_2_to_4_czech_sentences_of_its_outcome_and_situations(
    noted: list[Visit], crew: Crew
) -> None:
    silent = {e for e, share in crew.unrecorded.items() if share}
    notes = [v.note for v in noted if v.note]
    assert len(notes) >= 1_000
    for v in noted:
        if not v.recorded or v.tourplan.employee_id in silent:
            assert v.note is None
            continue
        assert v.note and 2 <= len(_sentences(v.note)) <= 4, v.note
        assert not SLOT.search(v.note)
    for note in notes[:1_000]:
        assert_czech(note, what="visit.note", allow=NAMES)
        assert has_diacritics(note)


def test_the_note_starts_with_a_sentence_of_the_outcome(noted: list[Visit], crew: Crew) -> None:
    book = crew.notebook
    for v in noted:
        if v.note:
            first = _sentences(v.note)[0]
            assert any(re.fullmatch(_pattern(s), first) for s in book.outcome(v)), (v.outcome, first)


def _pattern(sentence: str) -> str:
    parts = SLOT.split(sentence)
    return "".join(re.escape(part) if k % 2 == 0 else ".+" for k, part in enumerate(parts))


def _sales_visit(opening: Setup, crew: Crew, day: date) -> Visit:
    """A recorded sales visit of `day` that found the shop, not A-8's rep's."""
    world = opening.world
    return next(
        v
        for v in field.visit(world, crew, day, field.calls(world, crew, day, field.plan(world, day)))
        if v.recorded
        and v.kind == "sales"
        and v.outcome not in field.SHUT
        and v.tourplan.employee_id not in {e for e, s in crew.unrecorded.items() if s}
    )


def _says(note: str | None, sentences: tuple[str, ...]) -> bool:
    return note is not None and any(
        re.fullmatch(_pattern(s), sentence) for sentence in _sentences(note) for s in sentences
    )


def test_a_situation_met_has_its_sentence_a_planted_sighting_goes_in_verbatim(opening: Setup, crew: Crew) -> None:
    world = opening.world
    day = DAYS[0]
    v = _sales_visit(opening, crew, day)
    observations = field.observe(world, crew, day, [v])
    dry = [replace_row(r, out_of_stock=True) if k == 0 else r for k, r in enumerate(observations)]
    planted = field.Sighting(
        v.visit_id, "KAVALIR", "KAV-ZLATY-100", "promotion", 89.9, PLANTED_SIGHTINGS[1], True, planted=True
    )
    field.note(world, crew, day, [v], dry, [planted])
    assert v.note is not None and PLANTED_SIGHTINGS[1] in _sentences(v.note)
    book = crew.notebook
    assert _says(v.note, book.situations["out_of_stock"])
    background = replace(planted, description="Akce: Kavalír Zlatý za 89,90 Kč.", planted=False)
    v.note = None
    field.note(world, crew, day, [v], observations, [background])
    assert v.note is not None and any(
        re.fullmatch(_pattern(s), sentence) and "Kavalír" in sentence
        for sentence in _sentences(v.note)
        for s in book.situations["competitor_promotion"]
    )


def test_a13_s_first_refusal_is_said_plainly_and_the_later_ones_again(opening: Setup, crew: Crew) -> None:
    """review-233 R6: A-13's store has refused nothing before its first planted refusal."""
    world, day = opening.world, DAYS[2]
    v = _sales_visit(opening, crew, day)
    for again, said in ((False, (FIRST_REFUSAL,)), (True, PLANTED_REFUSALS)):
        v.note = None
        refusal = Listing("Nudle Kuřecí", False, "nemáme místo v regálu", planted=True, again=again)
        field.note(world, crew, day, [v], [], [], {v.visit_id: refusal})
        assert v.note is not None and any(sentence in said for sentence in _sentences(v.note)), v.note


def test_the_competitor_sentence_says_the_sighting_s_kind(opening: Setup, crew: Crew) -> None:
    """review-231 R3: a listing, a display or a price cut is never "a new promotion"."""
    world, day, book = opening.world, DAYS[1], crew.notebook
    v = _sales_visit(opening, crew, day)
    for kind in WORLD.field.sightings.kinds:
        seen = field.Sighting(v.visit_id, "KAVALIR", "KAV-ZLATY-100", kind, 89.9, "Kavalír Zlatý, 89,90 Kč.", True)
        v.note = None
        field.note(world, crew, day, [v], [], [seen])
        assert _says(v.note, book.situations[f"competitor_{kind}"]), (kind, v.note)
        others = [book.situations[f"competitor_{other}"] for other in WORLD.field.sightings.kinds if other != kind]
        assert not _says(v.note, tuple(s for sentences in others for s in sentences)), (kind, v.note)


def test_the_promotion_sentence_says_whether_the_store_executes_it(opening: Setup, crew: Crew) -> None:
    """review-231 R3: a promotion the observation finds not executed is never "on display"."""
    world, book = opening.world, crew.notebook
    said: Counter[bool] = Counter()
    for day in DAYS[:4]:
        visits = field.visit(world, crew, day, field.calls(world, crew, day, field.plan(world, day)))
        observations = field.observe(world, crew, day, visits)
        field.note(world, crew, day, visits, observations, [])
        first: dict[int, Row] = {}
        for r in observations:
            if r["promo_compliant"] is not None:
                first.setdefault(r["visit_id"], r)
        for v in visits:
            if v.note and v.visit_id in first:
                executed = bool(first[v.visit_id]["promo_compliant"])
                assert _says(v.note, book.situations["promotion_executed" if executed else "promotion_missing"])
                assert not _says(v.note, book.situations["promotion_missing" if executed else "promotion_executed"])
                assert executed or "je vystavená" not in v.note  # review-231: "displayed" where it is not
                said[executed] += 1
    assert said[True] > 20 and said[False] > 5


def test_a_visit_to_a_shop_closed_or_gone_saw_nothing_a_refused_one_stood_in_it(opening: Setup, crew: Crew) -> None:
    """review-231 R2: no shelf rows, no sighting, a note of two outcome sentences — even handed a refused visit's."""
    world, day, book = opening.world, DAYS[2], crew.notebook
    every = replace(crew, field=replace(crew.field, sightings=replace(crew.field.sightings, share=1.0)))
    refused = replace(_sales_visit(opening, crew, day), outcome="refused", note=None)
    rows, seen = field.observe(world, crew, day, [refused]), field.sight(world, every, day, [refused])
    field.note(world, crew, day, [refused], rows, seen)
    assert rows and seen and _says(refused.note, book.situations[f"competitor_{seen[0].kind}"])
    for outcome in field.SHUT:
        shut = replace(refused, outcome=outcome, note=None)
        assert field.observe(world, crew, day, [shut]) == [] and field.sight(world, every, day, [shut]) == []
        field.note(world, crew, day, [shut], rows, seen)
        assert shut.note is not None
        sentences = _sentences(shut.note)
        assert len(sentences) == 2 and all(_says(s, book.outcomes[outcome]) for s in sentences), shut.note


def test_acquisition_and_merchandising_notes_read_their_kind_s_sentences(noted: list[Visit], crew: Crew) -> None:
    """review-231 R3: a store never supplied, a station that orders through its key account — no sales sentence."""
    book = crew.notebook
    kinds: Counter[str] = Counter()
    for v in noted:
        if v.note and v.outcome == "no_order" and v.kind != "sales":
            assert _says(_sentences(v.note)[0], book.no_order_by_kind[v.kind]), v.note
            assert not _says(v.note, book.outcomes["no_order"]), v.note
            kinds[v.kind] += 1
    assert kinds["acquisition"] > 50 and kinds["merchandising"] > 5


def test_a_note_names_no_product_before_its_launch_or_after_its_delisting(noted: list[Visit], crew: Crew) -> None:
    """review-231 R3: e.g. Harcafé Gold Crema (2024-04-02) or Harcafé Latte (2024-10-01), not on sale in 2023."""
    off = {p.name for products in crew.products.values() for p in products if not any(p.sells(d) for d in DAYS)}
    assert {"Harcafé Gold Crema", "Harcafé Latte"} <= off
    for v in noted:
        assert v.note is None or not any(name in v.note for name in off), v.note
    gold = [p for products in crew.products.values() for p in products if p.name == "Harcafé Gold Crema"]
    assert [p.sells(date(2024, 4, 1)) for p in gold] == [False] and [p.sells(date(2024, 4, 2)) for p in gold] == [True]


def replace_row(row: dict[str, object], **values: object) -> dict[str, object]:
    return {**row, **values}


def test_the_notes_are_drawn_per_visit(opening: Setup, crew: Crew) -> None:
    world = opening.world
    day = DAYS[1]
    visits = field.visit(world, crew, day, field.calls(world, crew, day, field.plan(world, day)))
    twice = [replace(v) for v in visits]
    field.note(world, crew, day, visits, field.observe(world, crew, day, visits), [])
    field.note(world, crew, day, twice[::-1], field.observe(world, crew, day, twice), [])
    assert [v.note for v in visits] == [v.note for v in twice]
