"""NH-D37's paired S3 development verdict over saved CSV builds and a scratch PostgreSQL 16 database."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from hartland_sim.checks import s3
from hartland_sim.checks.s5_monthly_development import scratch_database, sha256, write_json
from hartland_sim.checks.stories import PROBES, Build, expected, holds
from hartland_sim.config import DATA_DIR, StoriesConfig, inventory
from hartland_sim.loop.build import BUILD_JSON
from hartland_sim.world import META_FILE, Checkpoint, Meta

LAST_DAY = date(2025, 12, 31)
WARM_UP_END = date(2023, 12, 31)
REP_COUNT = 50
STORY_MIN = 0.70
NULL_MAX = 0.30
TOLERANCE = 1e-9
FIELD_PROBES = ("A-6", "A-7", "A-8", "A-13")
DETECTOR = DATA_DIR / "stories/detectors/s3.sql"
QUERIES = DETECTOR.parent.parent / "s3_skill_queries.sql"
PERIOD_STARTS = ("2024-01-01", "2024-10-01", "2025-10-01")
PERIOD_ENDS = ("2024-09-30", "2025-09-30", "2025-12-31")
PINS = {"star", "weak"}


def correlation_passes(rho: float | None, null: bool) -> bool:
    return rho is not None and (abs(rho) < NULL_MAX if null else rho >= STORY_MIN)


def signed_correlations(
    rates: pd.DataFrame, labels: dict[str, dict[str, float]], null: bool
) -> dict[str, list[float | bool | None]]:
    skills = pd.DataFrame.from_dict(labels, orient="index")
    result: dict[str, list[float | bool | None]] = {}
    for measure in s3.MEASURES:
        observed = rates[measure]
        hidden = skills[measure].reindex(observed.index)
        rho = None
        if (
            len(observed) == REP_COUNT
            and observed.count() == hidden.count() == REP_COUNT
            and observed.nunique() > 1
            and hidden.nunique() > 1
        ):
            rho = statistics.correlation(
                observed.rank(method="average").tolist(), hidden.rank(method="average").tolist()
            )
        result[measure] = [rho, correlation_passes(rho, null) if rho is not None else None]
    return result


def read_labels(run: Path) -> dict[str, dict[str, float]]:
    actors = Checkpoint.load(run, WARM_UP_END).restore().actors
    values = {
        str(employee): {measure: getattr(actor.skills, measure) for measure in s3.MEASURES}
        for employee, actor in actors.items()
        if actor.role == "rep"
    }
    if len(values) != REP_COUNT or not all(math.isfinite(v) for row in values.values() for v in row.values()):
        raise ValueError(f"{run}: expected 50 employees with finite skill labels")
    return values


def warmup_slots(build: Build) -> list[list[int | str]]:
    warmup = s3.slots(build).sort_values(["employee_id", "code"])
    return [[int(employee), str(code)] for employee, code in warmup[["employee_id", "code"]].itertuples(index=False)]


def csv_readout(
    build: Build, stories: StoriesConfig, labels: dict[str, dict[str, float]], null: bool
) -> dict[str, Any]:
    observed = s3.counts(build, reconcile=True).sort_index()
    observed.index = observed.index.map(str)
    rates = s3.exact_rates(observed)
    ranking = rates.rank(method="min", ascending=False)
    rank_columns = list(s3.MEASURES)
    historical = rates.rename(columns={"retention": "churn"}).copy()
    historical["churn"] = -historical["churn"]
    historical["conversion"] = historical["conversion"].fillna(0)
    places = s3.ranks(historical)["place"]
    slots = warmup_slots(build)
    named = {pin.slot: pin.profile for pin in stories.pins.reps if pin.profile in PINS}
    profiles = {str(employee): named[str(code)] for employee, code in slots if code in named}
    assignments = build.narrow("assignment", ("employee_id", "area_id", "kind", "valid_from", "valid_to"))
    orders = build.narrow("sales_order", ("sales_order_id", "customer_id", "ordered_on", "channel"))
    last_order = orders["ordered_on"].max()
    duty = assignments[
        assignments["kind"].eq("rep_area")
        & assignments["valid_to"].ge(s3.HISTORY_FROM)
        & assignments["valid_from"].le(last_order)
    ].sort_values(["valid_from", "area_id"])
    periods = {
        str(employee): [
            [int(area), max(start, s3.HISTORY_FROM).date().isoformat(), min(end, last_order).date().isoformat()]
            for area, start, end in group[["area_id", "valid_from", "valid_to"]].itertuples(index=False)
        ]
        for employee, group in duty.groupby("employee_id")
    }
    rank_by_employee = ranking.to_dict(orient="index")
    reps = []
    for employee, row in observed.iterrows():
        key = str(employee)
        reps.append(
            {
                **{name: int(value) for name, value in row.items()},
                "employee_id": key,
                "place": int(places[key]),
                "profile": profiles.get(key),
                "periods": periods[key],
                "ranks": {m: int(rank_by_employee[key][m]) for m in rank_columns if pd.notna(rank_by_employee[key][m])}
                or None,
            }
        )
    probes = {case: int(PROBES[stories.anomaly(case).probe.sql](build, stories)) for case in FIELD_PROBES}
    return {
        "last_order": last_order.date().isoformat(),
        "slots": slots,
        "reps": reps,
        "correlations": signed_correlations(rates, labels, null),
        "probes": probes,
    }


def postgres_readout(
    dsn: str, run: Path, stories: StoriesConfig, labels: dict[str, dict[str, float]], null: bool
) -> dict[str, Any]:
    detector = DETECTOR.read_text(encoding="utf-8")
    ctes, end, _ = detector.rpartition("\nSELECT COUNT(*) FILTER (")
    if not end:
        raise ValueError("S3 detector has no placement select")
    statement = ctes + "\n" + QUERIES.read_text(encoding="utf-8")
    with (
        scratch_database(dsn) as database,
        psycopg.connect(database, autocommit=True, row_factory=dict_row) as conn,
        conn.transaction(force_rollback=True),
    ):
        if conn.info.server_version // 10_000 != 16:
            raise ValueError("S3 replay requires PostgreSQL 16")
        conn.execute("SET LOCAL search_path TO hartland")
        conn.execute("SET LOCAL work_mem = '96MB'")
        for table in inventory().tables:
            path = run / f"{table.name}.csv"
            if table.name.startswith("_sim_") or not path.is_file():
                continue
            with path.open(encoding="utf-8", newline="") as handle:
                header = next(csv.reader(handle))
            if set(header) != {column.name for column in table.columns}:
                raise ValueError(f"{path}: CSV columns differ from inventory")
            copy_sql = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT csv, HEADER true)").format(
                sql.Identifier(table.name), sql.SQL(", ").join(map(sql.Identifier, header))
            )
            with conn.cursor().copy(copy_sql) as copy, path.open("rb") as source:
                while chunk := source.read(1 << 20):
                    copy.write(chunk)
        conn.execute("SET CONSTRAINTS ALL IMMEDIATE")
        conn.execute("ANALYZE")
        conn.execute("REFRESH MATERIALIZED VIEW sales_pl")
        conn.execute("ANALYZE sales_pl")
        row = conn.execute(statement, {"labels": Jsonb(labels), "null": null}).fetchone()
        assert row is not None
        row["last_order"] = row["last_order"].isoformat()
        row["probes"] = {}
        for case in FIELD_PROBES:
            path = DATA_DIR / "stories" / stories.anomaly(case).probe.sql
            probe = conn.execute(path.read_text(encoding="utf-8")).fetchone()
            assert probe is not None
            row["probes"][case] = next(iter(probe.values()))
        metric = conn.execute(detector).fetchone()
        assert metric is not None
        row["legacy_metric"] = metric["metric"]
        row["postgres_version"] = conn.info.server_version
        return row


def legacy_places(rows: dict[str, Any]) -> dict[str, int]:
    by_employee = {str(employee): code for employee, code in rows["slots"]}
    return {
        by_employee[row["employee_id"]]: int(row["place"])
        for row in rows["reps"]
        if row["profile"] in PINS and row["employee_id"] in by_employee
    }


def validate(
    csv_rows: dict[str, Any],
    pg_rows: dict[str, Any],
    labels: dict[str, dict[str, float]],
    stories: StoriesConfig,
    null: bool,
) -> list[str]:
    failures = []
    for field in ("last_order", "slots", "reps", "probes"):
        if csv_rows[field] != pg_rows[field]:
            failures.append(f"CSV/PostgreSQL {field} differ")
    if csv_rows["last_order"] != LAST_DAY.isoformat():
        failures.append("incomplete order history")
    slots = csv_rows["slots"]
    ids = {str(employee) for employee, _ in slots}
    if (
        len(slots) != REP_COUNT
        or len(ids) != REP_COUNT
        or ids != set(labels)
        or len({slot for _, slot in slots}) != REP_COUNT
        or len(csv_rows["reps"]) != REP_COUNT
        or {row["employee_id"] for row in csv_rows["reps"]} != ids
    ):
        failures.append("employee identities or warm-up slot mapping differ")
    for row in csv_rows["reps"]:
        periods = row["periods"]
        if (
            tuple(period[1] for period in periods) != PERIOD_STARTS
            or tuple(period[2] for period in periods) != PERIOD_ENDS
            or any(a[0] == b[0] for a, b in zip(periods, periods[1:], strict=False))
        ):
            failures.append(f"{row['employee_id']}: territory history differs from the two registered moves")
        if any(row[exposure] <= 0 for _, exposure in s3.MEASURES.values()):
            failures.append(f"{row['employee_id']}: undefined exposure")
        if row["link_mismatches"] or row["linked_orders"] != row["converting_visits"]:
            failures.append(f"{row['employee_id']}: visit/order linkage mismatch")
    for measure, (rho, passed) in csv_rows["correlations"].items():
        pg_rho, pg_passed = (pg_rows["correlations"] or {}).get(measure, (None, None))
        if rho is None or pg_rho is None:
            failures.append(f"{measure}: undefined correlation")
        elif not math.isclose(float(rho), pg_rho, rel_tol=0, abs_tol=TOLERANCE) or pg_passed != passed:
            failures.append(f"{measure}: CSV/PostgreSQL correlation or decision differs")
        elif not passed:
            failures.append(f"{measure}: {'full-null control' if null else 'story threshold'} failed ({rho:.10f})")
    places = legacy_places(csv_rows)
    required = {pin.slot: pin.profile for pin in stories.pins.reps if pin.profile in PINS}
    if places.keys() != required.keys() or places != legacy_places(pg_rows):
        failures.append("historical named placements differ")
    else:
        n = sum(place <= 5 if required[slot] == "star" else place > REP_COUNT - 5 for slot, place in places.items())
        if n != pg_rows["legacy_metric"]:
            failures.append("historical placement verdict differs")
    for case, value in csv_rows["probes"].items():
        if value != 0 if null else not holds(value, *expected(stories.anomaly(case).probe.expect)):
            failures.append(f"{case}: anomaly probe failed ({value})")
    return failures


def read_pair(story: Path, full_null: Path, dsn: str, out: Path) -> int:
    stories = StoriesConfig.load()
    meta, control = Meta.load(story), Meta.load(full_null)
    if (
        meta.master_seed != control.master_seed
        or meta.world_sha256 != control.world_sha256
        or meta.last_day_done != LAST_DAY
        or control.last_day_done != LAST_DAY
        or meta.pilot_mode != "story"
        or control.pilot_mode != "full-null"
        or meta.stories_sha256 == control.stories_sha256
    ):
        raise ValueError("expected the same seed/source and full history in story and full-null modes")
    labels, null_labels = read_labels(story), read_labels(full_null)
    null_values = [value for row in null_labels.values() for value in row.values()]
    if labels.keys() != null_labels.keys() or any(value != 1.0 for value in null_values):
        raise ValueError("full-null employee mapping or constant skills differ")
    comparisons: dict[str, Any] = {}
    failures: list[str] = []
    for run, world, mode, null in ((story, meta, "story", False), (full_null, control, "full-null", True)):
        print(f"{world.master_seed} {world.pilot_mode}: reading CSV and PostgreSQL", flush=True)
        csv_rows = csv_readout(Build(run), stories, labels, null)
        pg_rows = postgres_readout(dsn, run, stories, labels, null)
        world_failures = validate(csv_rows, pg_rows, labels, stories, null)
        files = (BUILD_JSON, META_FILE, *(path.name for path in sorted(run.glob("*.csv"))))
        comparisons[mode] = {
            "build_id": world.build_id,
            "inputs": {name: sha256(run / name) for name in dict.fromkeys(files)},
            "csv": csv_rows,
            "postgres": pg_rows,
            "named_places": legacy_places(csv_rows),
            "failures": world_failures,
        }
        failures.extend(f"{world.pilot_mode}: {failure}" for failure in world_failures)
        write_json(out, {"seed": meta.master_seed, "comparisons": comparisons, "failures": failures})
    if comparisons["story"]["csv"]["slots"] != comparisons["full-null"]["csv"]["slots"]:
        failures.append("warm-up slot mapping differs between story and full null")
    write_json(
        out,
        {
            "seed": meta.master_seed,
            "comparisons": comparisons,
            "null_own_skill_correlations": {measure: None for measure in s3.MEASURES},
            "failures": failures,
            "passed": not failures,
        },
    )
    for failure in failures:
        print(failure)
    return int(bool(failures))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("story", type=Path)
    parser.add_argument("full_null", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--dsn", required=True, help="scratch PostgreSQL 16 server")
    args = parser.parse_args(argv)
    return read_pair(args.story, args.full_null, args.dsn, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
