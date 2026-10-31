"""G0 S0.2 T1 — the schema-checked CSV convention and `just reference-check`."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from hartland_sim.checks.reference import PER_FILE, check
from hartland_sim.cli import main
from hartland_sim.io.reference import ReferenceFileError, load_schema, read

GOOD_CSV = "day,qty,name,ok,note\n2024-01-01,3,Kolín,true,\n2024-01-02,4,Hodonín,false,něco\n"

# Every reference file of the repo, by name: one that disappears or moves fails. A stage that adds files adds them.
COMMITTED = (
    "area",
    "area_okres",
    "calendar",
    "cost_centre",
    "gl_account",
    "kraj",
    "legal_entity",
    "names_first",
    "names_last",
    "okres",
    "region",
    "severe_days",
    "site",
    "wage_by_kraj",
    "wage_by_okres",
    "weather_normals",
)


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


def town_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "x-rows": [1, 9],
        "type": "object",
        "additionalProperties": False,
        "required": ["name"],
        "properties": {"name": {"type": "string"}},
    }


def reference_dir(tmp_path: Path, files: dict[str, tuple[dict[str, Any] | str | None, str | None]]) -> Path:
    """A reference folder: per name, its schema (a dict, or the file's raw text) and its CSV text."""
    (tmp_path / "schemas").mkdir()
    for name, (schema, text) in files.items():
        if schema is not None:
            raw = schema if isinstance(schema, str) else json.dumps(schema)
            (tmp_path / "schemas" / f"{name}.schema.json").write_text(raw, encoding="utf-8")
        if text is not None:
            (tmp_path / f"{name}.csv").write_text(text, encoding="utf-8", newline="")
    return tmp_path


def problems(tmp_path: Path, text: str, **extra: Any) -> list[str]:
    return check(reference_dir(tmp_path, {"mini": (mini_schema(**extra), text)})).problems


def test_the_committed_reference_files_hold(repo_root: Path) -> None:
    root = repo_root / "data" / "reference"
    report = check(root)
    assert report.problems == []
    assert sorted(path.stem for path in root.glob("*.csv")) == list(COMMITTED)
    assert report.files == len(COMMITTED)


def test_the_default_folder_is_the_repos_wherever_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert read("region")[0]["code"] == "PHA"
    assert main(["reference-check"]) == 0
    assert f"{len(COMMITTED)} files" in capsys.readouterr().out


@pytest.mark.parametrize("folder", ["missing", "empty"])
def test_a_folder_without_reference_files_fails(
    tmp_path: Path, folder: str, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / folder
    if folder == "empty":
        (root / "schemas").mkdir(parents=True)
    assert check(root).problems == [f"{root}: no reference files (is --dir right?)"]
    assert main(["reference-check", "--dir", str(root)]) == 1
    assert "1 problem" in capsys.readouterr().err


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
    assert found[0].startswith("mini.csv line 4, column `name`: English words ['the']")


def test_each_english_value_is_named_by_its_line(tmp_path: Path) -> None:
    """A sparse column: the empty cells do not shift the lines, and every English value is reported."""
    schema = mini_schema()
    schema["properties"]["note"]["x-czech"] = True
    rows = "2024-01-03,3,Kolín,true,The Day\n2024-01-04,3,Kolín,true,\n2024-01-05,3,Kolín,true,Svátek of the Free\n"
    found = check(reference_dir(tmp_path, {"mini": (schema, GOOD_CSV + rows)})).problems
    assert [problem.split(": English words")[0] for problem in found] == [
        "mini.csv line 4, column `note`",
        "mini.csv line 6, column `note`",
    ]


def test_a_value_that_is_not_nfc_is_one_problem(tmp_path: Path) -> None:
    assert problems(tmp_path, GOOD_CSV + "2024-01-03,3,Koli\u0301n,true,\n") == [
        "mini.csv line 4, column `name`: not NFC"
    ]


def test_a_column_of_english_labels_fails_as_a_whole(tmp_path: Path) -> None:
    """Each label alone is too short to tell; the column together is not."""
    rows = "".join(f"2024-02-{day:02},3,Big Coffee Shop {day},true,\n" for day in range(1, 13))
    found = problems(tmp_path, "day,qty,name,ok,note\n" + rows, **{"x-rows": [1, 20]})
    assert found == ["mini.csv, column `name`: 12 texts, 218 characters without a Czech diacritic"]


def test_the_file_format_is_checked(tmp_path: Path) -> None:
    found = problems(tmp_path, "﻿" + GOOD_CSV.replace("\n", "\r\n").rstrip("\r\n"))
    assert "mini.csv: starts with a byte-order mark" in found
    assert "mini.csv: CR line ends (the convention is \\n)" in found
    assert "mini.csv: no newline at the end" in found


def test_a_reference_must_resolve(tmp_path: Path) -> None:
    schema = mini_schema()
    schema["properties"]["name"]["x-ref"] = "town.name"
    root = reference_dir(tmp_path, {"mini": (schema, GOOD_CSV), "town": (town_schema(), "name\nKolín\n")})
    assert check(root).problems == ["mini.csv line 3, column `name`: 'Hodonín' is not in town.name"]


def test_a_reference_into_a_file_that_did_not_load_is_one_line(tmp_path: Path) -> None:
    schema = mini_schema()
    schema["properties"]["name"]["x-ref"] = "town.name"
    root = reference_dir(tmp_path, {"mini": (schema, GOOD_CSV), "town": (town_schema(), "nm\nKolín\n")})
    assert check(root).problems == [
        "mini.csv, column `name`: not checked — town.csv did not load",
        "town.csv line 1: header ['nm'] ≠ the schema's columns ['name']",
    ]


def test_a_row_that_does_not_parse_still_counts(tmp_path: Path) -> None:
    found = problems(tmp_path, GOOD_CSV + "2024-01-03,x,Kolín,true,\n", **{"x-rows": [3, 3]})
    assert found == ["mini.csv line 4, column `qty`: 'x' is not integer"]


@pytest.mark.parametrize(
    ("schema", "text", "says"),
    [
        (mini_schema(), GOOD_CSV + '2024-01-03,3,"Kolín\n', "mini.csv line 4: unexpected end of data"),
        ('{"x-rows":', GOOD_CSV, "schemas/mini.schema.json: not JSON (Expecting value"),
        ("[]", GOOD_CSV, "schemas/mini.schema.json: not a JSON object"),
        ({"x-rows": [1, 2]}, GOOD_CSV, "schemas/mini.schema.json: the convention wants `properties`"),
        (
            mini_schema(properties=mini_schema()["properties"] | {"qty": {"type": "integer", "x-czech": True}}),
            GOOD_CSV,
            "schemas/mini.schema.json: the convention wants `x-czech` on a string column only (`qty`)",
        ),
    ],
    ids=["unclosed-quote", "not-json", "not-an-object", "no-properties", "x-czech-on-a-number"],
)
def test_a_malformed_file_is_a_problem_not_a_crash(
    tmp_path: Path, schema: dict[str, Any] | str, text: str, says: str
) -> None:
    report = check(reference_dir(tmp_path, {"mini": (schema, text), "town": (town_schema(), "name\nKolín\n")}))
    assert any(problem.startswith(says) for problem in report.problems), report.problems
    assert not any(problem.startswith("town.csv") for problem in report.problems)
    assert report.files == 2


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


@pytest.mark.parametrize(
    ("keyword", "value", "wants"),
    [
        ("x-rows", [True, True], "`x-rows: [min, max]`"),
        ("x-rows", [5, 1], "`x-rows: [min, max]`"),
        ("x-key", "day", "`x-key` names columns"),
        ("x-key", ["date"], "`x-key` names columns"),
    ],
)
def test_the_row_keywords_are_checked(tmp_path: Path, keyword: str, value: object, wants: str) -> None:
    found = problems(tmp_path, GOOD_CSV, **{keyword: value})
    assert found == [f"schemas/mini.schema.json: the convention wants {wants}"]


@pytest.mark.parametrize("target", ["town", "town.name.x", 7])
def test_an_x_ref_names_a_file_and_a_column(tmp_path: Path, target: object) -> None:
    schema = mini_schema()
    schema["properties"]["name"]["x-ref"] = target
    found = check(reference_dir(tmp_path, {"mini": (schema, GOOD_CSV)})).problems
    assert found == ['schemas/mini.schema.json: the convention wants `x-ref: "<file>.<column>"` (`name`)']


def test_read_types_the_cells(tmp_path: Path) -> None:
    rows = read("mini", reference_dir(tmp_path, {"mini": (mini_schema(), GOOD_CSV)}))
    assert rows[0] == {"day": date(2024, 1, 1), "qty": 3, "name": "Kolín", "ok": True, "note": None}
    assert rows[1]["note"] == "něco"


def test_read_names_the_bad_column(tmp_path: Path) -> None:
    root = reference_dir(tmp_path, {"mini": (mini_schema(), GOOD_CSV + "2024-01-03,x,Kolín,true,\n")})
    with pytest.raises(ReferenceFileError, match=r"mini.csv line 4, column `qty`: 'x' is not integer"):
        read("mini", root)


@pytest.mark.parametrize(
    ("text", "says"),
    [
        (GOOD_CSV.replace("\n", "\r\n"), r"mini.csv: CR line ends"),
        (GOOD_CSV + "2024-01-03,3,Kolín ,true,\n", r"mini.csv line 4, column `name`: whitespace"),
        (GOOD_CSV + "2024-01-03,3,Koli\u0301n,true,\n", r"mini.csv line 4, column `name`: not NFC"),
        (GOOD_CSV + '2024-01-03,3,"Kolín\n', r"mini.csv line 4: unexpected end of data"),
    ],
    ids=["crlf", "whitespace", "nfd", "unclosed-quote"],
)
def test_read_holds_the_file_to_the_same_convention(tmp_path: Path, text: str, says: str) -> None:
    with pytest.raises(ReferenceFileError, match=says):
        read("mini", reference_dir(tmp_path, {"mini": (mini_schema(), text)}))


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


def test_the_verb_counts_every_problem_not_the_lines_shown(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows = "".join(f"2024-02-{day:02},x,Kolín,true,\n" for day in range(1, PER_FILE + 6))
    root = reference_dir(tmp_path, {"mini": (mini_schema(**{"x-rows": [1, 30]}), GOOD_CSV + rows)})
    assert main(["reference-check", "--dir", str(root)]) == 1
    err = capsys.readouterr().err.splitlines()
    assert err[-2:] == ["mini.csv: … and 5 more", f"hartland-sim reference-check: {PER_FILE + 5} problems"]
