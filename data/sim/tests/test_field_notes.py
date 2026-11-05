"""The visit notes (G2 S2.1 T6, `loop.field.note`): a recorded visit's note is a sentence of its outcome from
`visit_note_templates.yaml` and one of each situation it met — a mandatory promotion running, a line out of stock, a
competitor seen, a display lent — 2–4 sentences, their slots filled from the visit (a SKU's name or nickname, the
competitor, the promotion, an objection), stream `note`; A-6's sightings go in hand-written and verbatim; an unrecorded
visit and every visit of A-8's rep have none. The visits are the opening `World`'s over two weeks of 2023, their
outcomes as drawn and the order ones set by hand."""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import date, timedelta

import pytest

from hartland_sim.checks.czech import assert_czech, has_diacritics
from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.io.reference import read
from hartland_sim.loop import field
from hartland_sim.loop.field import PLANTED_SIGHTINGS, Crew, Visit
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
            assert any(re.fullmatch(_pattern(s), first) for s in book.outcomes[v.outcome]), (v.outcome, first)


def _pattern(sentence: str) -> str:
    parts = SLOT.split(sentence)
    return "".join(re.escape(part) if k % 2 == 0 else ".+" for k, part in enumerate(parts))


def test_a_situation_met_has_its_sentence_a_planted_sighting_goes_in_verbatim(opening: Setup, crew: Crew) -> None:
    world = opening.world
    day = DAYS[0]
    v = next(
        v
        for v in field.visit(world, crew, day, field.calls(world, crew, day, field.plan(world, day)))
        if v.recorded
        and v.kind == "sales"
        and v.tourplan.employee_id not in {e for e, s in crew.unrecorded.items() if s}
    )
    observations = field.observe(world, crew, day, [v])
    dry = [replace_row(r, out_of_stock=True) if k == 0 else r for k, r in enumerate(observations)]
    planted = field.Sighting(v.visit_id, "KAVALIR", "KAV-ZLATY-100", "promotion", 89.9, PLANTED_SIGHTINGS[1], True)
    field.note(world, crew, day, [v], dry, [planted])
    assert v.note is not None and PLANTED_SIGHTINGS[1] in _sentences(v.note)
    book = crew.notebook
    assert any(
        re.fullmatch(_pattern(s), sentence) for sentence in _sentences(v.note) for s in book.situations["out_of_stock"]
    )
    background = replace(planted, description="Akce: Kavalír Zlatý za 89,90 Kč.")
    v.note = None
    field.note(world, crew, day, [v], observations, [background])
    assert v.note is not None and any(
        re.fullmatch(_pattern(s), sentence) and "Kavalír" in sentence
        for sentence in _sentences(v.note)
        for s in book.situations["competitor"]
    )


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
