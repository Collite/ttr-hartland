"""G0 S0.2 T1 — the schema-checked CSV convention and `just reference-check`."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from hartland_sim.checks.reference import check
from hartland_sim.cli import main
from hartland_sim.io.reference import ReferenceFileError, load_schema, read

GOOD_CSV = "day,qty,name,ok,note\n2024-01-01,3,Kolín,true,\n2024-01-02,4,Hodonín,false,něco\n"


def mini_schema(**extra: Any) -> dict[str, Any]:
    columns = {
        "day": {"type": "string", "format": "date", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
        "qty": {"type": "integer", "minimum": 0},
        "name": {"type": "string", "minLength": 1, "x-czech": True},
        "ok": {"type": "boolean"},
        "note": {"type": ["string", "null"]},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "x-rows": [1, 10],
        "x-key": ["day"],
        "type": "object",
        "additionalProperties": False,
        "required": list(columns),
        "properties": columns,
    } | extra


def reference_dir(tmp_path: Path, files: dict[str, tuple[dict[str, Any] | None, str | None]]) -> Path:
    (tmp_path / "schemas").mkdir()
    for name, (schema, text) in files.items():
        if schema is not None:
            (tmp_path / "schemas" / f"{name}.schema.json").write_text(json.dumps(schema), encoding="utf-8")
        if text is not None:
            (tmp_path / f"{name}.csv").write_text(text, encoding="utf-8", newline="")
    return tmp_path


def problems(tmp_path: Path, text: str, **extra: Any) -> list[str]:
    return check(reference_dir(tmp_path, {"mini": (mini_schema(**extra), text)})).problems


def test_the_committed_reference_files_hold(repo_root: Path) -> None:
    root = repo_root / "data" / "reference"
    report = check(root)
    assert report.problems == []
    assert report.files == len(list(root.glob("*.csv")))


def test_a_good_file_passes(tmp_path: Path) -> None:
    assert problems(tmp_path, GOOD_CSV) == []


@pytest.mark.parametrize(
    ("row", "column", "says"),
    [
        ("2024-02-30,3,Kolín,true,", "day", "is not a 'date'"),
        ("2024-01-03,-1,Kolín,true,", "qty", "less than the minimum"),
        ("2024-01-03,tři,Kolín,true,", "qty", "'tři' is not integer"),
        ("2024-01-03,,Kolín,true,", "qty", "not nullable"),
        ("2024-01-03,3,Kolín,ano,", "ok", "'ano' is not boolean"),
        ("2024-01-03,3,,true,", "name", "should be non-empty"),
        ("2024-01-03,3, Kolín,true,", "name", "whitespace"),
        ("2024-01-03,3,Koli\u0301n,true,", "name", "not NFC"),
    ],
)
def test_a_broken_row_fails_with_the_column_named(tmp_path: Path, row: str, column: str, says: str) -> None:
    found = problems(tmp_path, GOOD_CSV + row + "\n")
    assert any(f"mini.csv line 4, column `{column}`" in p and says in p for p in found), found


def test_a_wrong_header_fails(tmp_path: Path) -> None:
    found = problems(tmp_path, GOOD_CSV.replace("qty", "quantity"))
    assert found == [
        "mini.csv line 1: header ['day', 'quantity', 'name', 'ok', 'note'] "
        "≠ the schema's columns ['day', 'qty', 'name', 'ok', 'note']"
    ]


def test_a_short_row_fails(tmp_path: Path) -> None:
    assert problems(tmp_path, GOOD_CSV + "2024-01-03,3\n") == ["mini.csv line 4: 2 cells, the header has 5"]


def test_the_row_count_is_checked(tmp_path: Path) -> None:
    assert problems(tmp_path, GOOD_CSV, **{"x-rows": [5, 9]}) == ["mini.csv: 2 rows, expected 5–9"]


def test_a_duplicate_key_fails(tmp_path: Path) -> None:
    found = problems(tmp_path, GOOD_CSV + "2024-01-01,9,Kolín,true,\n")
    assert found == ["mini.csv line 4: key (day) = ('2024-01-01',) again (first on line 2)"]


def test_english_in_a_czech_column_fails(tmp_path: Path) -> None:
    found = problems(tmp_path, GOOD_CSV + "2024-01-03,3,The Factory,true,\n")
    assert len(found) == 1
    assert found[0].startswith("mini.csv, column `name`[2]: English words ['the']")


def test_the_file_format_is_checked(tmp_path: Path) -> None:
    found = problems(tmp_path, "﻿" + GOOD_CSV.replace("\n", "\r\n").rstrip("\r\n"))
    assert "mini.csv: starts with a byte-order mark" in found
    assert "mini.csv: CR line ends (the convention is \\n)" in found
    assert "mini.csv: no newline at the end" in found


def test_a_reference_must_resolve(tmp_path: Path) -> None:
    town = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "x-rows": [1, 9],
        "type": "object",
        "additionalProperties": False,
        "required": ["name"],
        "properties": {"name": {"type": "string"}},
    }
    schema = mini_schema()
    schema["properties"]["name"]["x-ref"] = "town.name"
    root = reference_dir(tmp_path, {"mini": (schema, GOOD_CSV), "town": (town, "name\nKolín\n")})
    assert check(root).problems == ["mini.csv line 3, column `name`: 'Hodonín' is not in town.name"]


def test_every_csv_has_a_schema_and_every_schema_a_csv(tmp_path: Path) -> None:
    root = reference_dir(tmp_path, {"orphan": (None, GOOD_CSV), "ghost": (mini_schema(), None)})
    assert check(root).problems == [
        "orphan.csv: no schema (schemas/orphan.schema.json)",
        "schemas/ghost.schema.json: no ghost.csv",
    ]


def test_a_schema_must_follow_the_convention(tmp_path: Path) -> None:
    schema = mini_schema(required=["day"])
    del schema["x-rows"]
    found = check(reference_dir(tmp_path, {"mini": (schema, GOOD_CSV)})).problems
    assert found == [
        "schemas/mini.schema.json: the convention wants `required` lists every column, in order",
        "schemas/mini.schema.json: the convention wants `x-rows: [min, max]`",
    ]


def test_read_types_the_cells(tmp_path: Path) -> None:
    rows = read("mini", reference_dir(tmp_path, {"mini": (mini_schema(), GOOD_CSV)}))
    assert rows[0] == {"day": date(2024, 1, 1), "qty": 3, "name": "Kolín", "ok": True, "note": None}
    assert rows[1]["note"] == "něco"


def test_read_names_the_bad_column(tmp_path: Path) -> None:
    root = reference_dir(tmp_path, {"mini": (mini_schema(), GOOD_CSV + "2024-01-03,x,Kolín,true,\n")})
    with pytest.raises(ReferenceFileError, match=r"mini.csv line 4, column `qty`: 'x' is not integer"):
        read("mini", root)


LABELS = ("name", "town", "nominative", "vocative", "genitive", "male", "female")


def test_every_label_column_is_guarded(repo_root: Path) -> None:
    """DONE: no English anywhere — `assert_czech_all` on each label column (review-188 R2)."""
    for path in sorted((repo_root / "data" / "reference" / "schemas").glob("*.schema.json")):
        for column, spec in load_schema(path)["properties"].items():
            if column in LABELS or column.endswith("_cs"):
                assert spec.get("x-czech") is True, f"{path.name}: {column}"


def test_the_verb_passes_on_the_committed_files(repo_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["reference-check", "--dir", str(repo_root / "data" / "reference")]) == 0
    assert capsys.readouterr().out.endswith("rows — OK\n")


def test_the_verb_fails_listing_the_problems(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = reference_dir(tmp_path, {"mini": (mini_schema(), GOOD_CSV + "2024-02-30,3,Kolín,true,\n")})
    assert main(["reference-check", "--dir", str(root)]) == 1
    err = capsys.readouterr().err
    assert "mini.csv line 4, column `day`" in err
    assert "hartland-sim reference-check: 1 problem" in err
