"""S5's full no-PoS reference and the development gate (G2 S2.7 T3–T5; NH-D21-B design08 §§7–8).

**The replay** of a scored pilot build is the no-PoS pilot build (`PAIRS`: `no-pos` for the story and the
skill-neutral control, `no-pos-null` for the full null) of the same seed and source. S2.6's modes run every day from
the warm-up's first, the PoS-free months included, under the same keyed draws and the story pre-pilot's assignment map,
so the two builds part only where the programme acts: no display, no give-away, no display effect in the replay, and
its demand, orders and prices run whole.

**Isolation** (`isolation`; scope as Bora ruled on 2026-11-17, S2.7's T3 Found): every CSV of the actual build
against the replay's, before anything is scored. The treatment may differ: the PoS tables (`POS_TABLES`, and the PoS
ledger's `MAJ-POS-` rows of `asset_register`) and every row of a programme customer, found by its customer_id, its
order's, its order line's or its visit's. The prompt rulebook's figures are channel-wide, so a holdout's rep prompts
and visit notes may differ too: `rep_prompt`'s rows and `visit.note` are reported, by arm, never a failure. Any other
difference — a holdout's or an outside customer's order, line or discount, a visit's other columns, a budget, a plan,
an agreement, a row only one side has, a file only one side has — fails, named by its table and the arms it touches:
a spillover's causal path to trace, never one to repair here (design08 §7).

**The reference** (`reference`): per region, the programme customers' whole Customer Margin in W in the actual build
less in the replay (`reference_gain`: every line, the displayed one and the others alike, never a line's margin
divided by its uplift), and how far the holdouts' moved (`holdout_change`, `holdouts_changed`: 0 where isolation
holds). The cost is the actual build's. The observed gain's normalized error is (gain − reference) ÷ max(|reference|,
cost) (`errors`); the truth-based verdict judges the observed frame with the reference's gains (`truth`).

**The development gate** (`develop`; `python -m hartland_sim.checks.s5_replay ROOT`): the protocol's three development
seeds × its five modes under `ROOT/<seed>/<mode>/`, each the seed and the mode its folder names, all at one clean
source (`build.json`'s git sha) on one world.yaml and this protocol, one assignment digest per seed, each done to W's
end. Isolation runs first on every scored build and its replay: one failure is NO-GO and nothing is scored. Then
each scored build is judged (`s5_pilot.judge`), referenced and judged again on the reference. GO when every story
passes, every control (skill-neutral, full-null) is REJECTED with both halves failing — rank below the bar and the
group claim failing — every truth-based verdict is the observed one, and in every scored mode each region's normalized
errors over the three seeds have |mean| ≤ the protocol's `bias` and RMS ≤ its `rms`. Anything else is NO-GO, never
a pass on the reference alone; a development GO is no certification (S2.8's). The readout (`s5-development.json` and
`.md` in ROOT) records every build's identity, the isolation, frames, verdicts, references and errors, and the
placement decisions' outcomes and forecasts (the policy's components, reported, never judged).
"""

from __future__ import annotations

import csv
import filecmp
import json
import math
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hartland_sim.checks import s5_pilot
from hartland_sim.checks.stories import Build
from hartland_sim.protocol import Protocol

Frame = pd.DataFrame

PAIRS = {"story": "no-pos", "skill-neutral": "no-pos", "full-null": "no-pos-null"}  # a scored mode → its replay
OWN = ("_sim_meta.csv", "_sim_checkpoint.csv")  # the generator's, never compared
POS_TABLES = ("pos_loan", "pos_handout", "pos_asset", "pos_placement_decision")
POS_REGISTER = "MAJ-POS-"  # `loop.pos.ledger`'s asset_register codes: the PoS assets in aggregate
READOUT = "s5-development"

csv.field_size_limit(1 << 30)


# ── isolation ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Isolation:
    """`isolation`'s result: the failures (empty when clean), the reported differences and the treatment's, by table
    and arm."""

    actual: str
    replay: str
    failures: tuple[str, ...]
    reported: dict[str, int]
    treatment: dict[str, int]

    @property
    def clean(self) -> bool:
        return not self.failures


def _rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        yield from csv.DictReader(source)


def _hashes(path: Path) -> np.ndarray[Any, Any]:
    with path.open("rb") as source:
        next(source, None)
        return np.fromiter((hash(line) for line in source), dtype=np.int64)


def _differing(path: Path, only: set[int]) -> list[dict[str, str]]:
    """The rows of `path` whose lines hash into `only`."""
    with path.open("rb") as source:
        header = next(source).decode("utf-8")
        kept = [line.decode("utf-8") for line in source if hash(line) in only]
    return list(csv.DictReader([header, *kept]))


def isolation(actual: Path, replay: Path) -> Isolation:
    """`actual` against its replay (module docstring)."""
    arm = {row["customer_id"]: row["arm"] for row in _rows(actual / "pos_pilot_assignment.csv")}
    order_of: dict[str, str] = {}
    visit_of: dict[str, str] = {}
    for run in (actual, replay):
        order_of |= {row["sales_order_id"]: row["customer_id"] for row in _rows(run / "sales_order.csv")}
        visit_of |= {row["visit_id"]: row["customer_id"] for row in _rows(run / "visit.csv")}
    failures: list[str] = []
    reported: Counter[str] = Counter()
    treatment: Counter[str] = Counter()
    names = sorted({p.name for p in actual.glob("*.csv")} | {p.name for p in replay.glob("*.csv")})
    for name in names:
        table = name.removesuffix(".csv")
        mine, theirs = actual / name, replay / name
        if name in OWN:
            continue
        if not (mine.is_file() and theirs.is_file()):
            failures.append(f"{table}: only in the {'actual build' if mine.is_file() else 'replay'}")
            continue
        if filecmp.cmp(mine, theirs, shallow=False):
            continue
        ours, other = _hashes(mine), _hashes(theirs)
        if table in POS_TABLES:
            treatment[table] += len(np.setdiff1d(ours, other)) + len(np.setdiff1d(other, ours))
            continue
        sides = {
            "actual": _differing(mine, set(np.setdiff1d(ours, other).tolist())),
            "replay": _differing(theirs, set(np.setdiff1d(other, ours).tolist())),
        }
        line_order = _line_orders(table, sides, actual, replay)
        kept: Counter[str] = Counter()
        visits: dict[str, dict[str, dict[str, str]]] = {"actual": {}, "replay": {}}
        for side, rows in sides.items():
            for row in rows:
                who = _arm(row, arm, order_of, visit_of, line_order)
                if (table == "asset_register" and row.get("code", "").startswith(POS_REGISTER)) or who == "programme":
                    treatment[table] += 1
                elif table == "rep_prompt":
                    reported[f"rep_prompt:{who}"] += 1
                elif table == "visit":
                    visits[side][row["visit_id"]] = row | {"_arm": who}
                else:
                    kept[who] += 1
        for visit_id in visits["actual"].keys() | visits["replay"].keys():
            one, two = visits["actual"].get(visit_id), visits["replay"].get(visit_id)
            if one is not None and two is not None and {k for k in one if one[k] != two[k]} == {"note"}:
                reported[f"visit.note:{one['_arm']}"] += 1
            else:
                kept[(one or two or {}).get("_arm", "-")] += 1
        if kept:
            failures.append(
                f"{table}: {sum(kept.values())} rows differ ({', '.join(f'{k} {n}' for k, n in sorted(kept.items()))})"
            )
    return Isolation(
        str(actual), str(replay), tuple(failures), dict(sorted(reported.items())), dict(sorted(treatment.items()))
    )


def _line_orders(table: str, sides: dict[str, list[dict[str, str]]], *runs: Path) -> dict[str, str]:
    """For a table keyed by order line only, the differing rows' lines' orders (streamed from both builds)."""
    wanted = {row["sales_order_line_id"] for rows in sides.values() for row in rows if "sales_order_line_id" in row}
    if not wanted or table == "sales_order_line":
        return {}
    out: dict[str, str] = {}
    for run in runs:
        for row in _rows(run / "sales_order_line.csv"):
            if row["sales_order_line_id"] in wanted:
                out[row["sales_order_line_id"]] = row["sales_order_id"]
    return out


def _arm(
    row: dict[str, str],
    arm: dict[str, str],
    order_of: dict[str, str],
    visit_of: dict[str, str],
    line_order: dict[str, str],
) -> str:
    """The pilot arm of the customer a row belongs to: `other` for a customer in no arm, `-` for a row of none."""
    customer = row.get("customer_id") or ""
    if not customer and row.get("sales_order_id"):
        customer = order_of.get(row["sales_order_id"], "")
    if not customer and row.get("sales_order_line_id"):
        customer = order_of.get(line_order.get(row["sales_order_line_id"], ""), "")
    if not customer and row.get("visit_id"):
        customer = visit_of.get(row["visit_id"], "")
    if not customer:
        return "-"
    return arm.get(customer, "other")


# ── the reference ──────────────────────────────────────────────────────────────


def reference(actual: Frame, replay: Frame, codes: dict[Any, str]) -> Frame:
    """Per region code, from the actual build's and the replay's `s5_pilot.outcomes` (module docstring):
    `reference_gain` and `holdout_change` in CZK and `holdouts_changed`; `ValueError` when their paired customers or
    arms differ."""
    actual, replay = actual.sort_index(), replay.sort_index()
    if not actual.index.equals(replay.index) or not actual["arm"].equals(replay["arm"]):
        raise ValueError("the build and its replay pair different customers")
    moved = (actual["margin"] - replay["margin"]).astype(np.int64)
    frame = pd.DataFrame({"region_id": actual["region_id"], "arm": actual["arm"], "moved": moved})
    out = []
    for region_id, code in codes.items():
        mine = frame[frame["region_id"] == region_id]
        programme, holdout = mine[mine["arm"] == "programme"], mine[mine["arm"] == "holdout"]
        out.append(
            {
                "region": code,
                "reference_gain": int(programme["moved"].sum()) / 100,
                "holdout_change": int(holdout["moved"].abs().sum()) / 100,
                "holdouts_changed": int((holdout["moved"] != 0).sum()),
            }
        )
    return pd.DataFrame(out).set_index("region")


def errors(observed: Frame, truth: Frame) -> pd.Series[float]:
    """Each region's normalized error, (gain − reference) ÷ max(|reference|, cost)."""
    scale = np.maximum(truth["reference_gain"].abs(), observed["cost"])
    return (observed["gain"] - truth["reference_gain"]) / scale


def truth(observed: Frame, reference_frame: Frame) -> Frame:
    """The observed frame with the reference's gains: what the truth-based verdict judges."""
    out = observed.copy()
    out["gain"] = reference_frame["reference_gain"].reindex(out.index)
    out["gain_per_cost"] = (out["gain"] / out["cost"]).where(out["cost"] > 0)
    out["net_return"] = out["gain_per_cost"] - 1
    return out


def components(build: Build) -> Frame:
    """The policy's components by region code (reported): the placement decisions by outcome, and the selected
    ones' forecast gain and cost (their pre-decision snapshots)."""
    decisions = build["pos_placement_decision"]
    codes = build["region"].set_index("region_id")["code"]
    decisions = decisions.assign(region=decisions["region_id"].map(codes))
    out = decisions.pivot_table(index="region", columns="outcome", values="customer_id", aggfunc="count", fill_value=0)
    selected = decisions[decisions["outcome"] == "selected"].groupby("region")
    out["forecast_gain"] = selected["forecast_gain"].sum()
    out["forecast_cost"] = selected["forecast_cost"].sum()
    return out.fillna(0)


# ── the development gate ───────────────────────────────────────────────────────


@dataclass
class Scored:
    """One scored build's results."""

    seed: int
    mode: str
    status: str
    reasons: list[str]
    verdict: dict[str, Any] | None
    truth: dict[str, Any] | None
    frame: dict[str, Any] | None
    reference: dict[str, Any] | None
    errors: dict[str, float] | None
    components: dict[str, Any] | None
    later_acquired: int


@dataclass
class Readout:
    """`develop`'s readout (module docstring)."""

    root: str
    decision: str  # GO · NO-GO
    reasons: list[str]
    identity: dict[str, dict[str, Any]]
    isolation: list[dict[str, Any]]
    scored: list[Scored] = field(default_factory=list)
    bias: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)


def _run(root: Path, seed: int, mode: str) -> Path:
    return root / str(seed) / mode


def _records(frame: Frame) -> dict[str, Any]:
    return {str(k): {c: _plain(v) for c, v in row.items()} for k, row in frame.to_dict(orient="index").items()}


def _plain(value: Any) -> Any:
    if isinstance(value, float | np.floating):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def develop(root: Path, protocol: Protocol | None = None) -> Readout:
    """The development gate over ROOT's builds (module docstring)."""
    protocol = protocol or Protocol.load()
    seeds, modes = protocol.development, protocol.modes
    identity = {f"{seed}/{mode}": s5_pilot.identity(_run(root, seed, mode)) for seed in seeds for mode in modes}
    wrong = _identity(identity, protocol)
    readout = Readout(str(root), "NO-GO", wrong, identity, [])
    if wrong:
        return readout
    for seed in seeds:  # isolation first: nothing is scored unless every replay holds
        for mode in s5_pilot.SCORED:
            held = isolation(_run(root, seed, mode), _run(root, seed, PAIRS[mode]))
            readout.isolation.append({"seed": seed, "mode": mode, "clean": held.clean} | asdict(held))
            readout.reasons += [f"{seed}/{mode} isolation: {failure}" for failure in held.failures]
    if readout.reasons:
        return readout
    for seed in seeds:
        replays: dict[str, Frame] = {}
        for mode in s5_pilot.SCORED:
            replay = PAIRS[mode]
            if replay not in replays:
                replays[replay] = s5_pilot.outcomes(Build(_run(root, seed, replay)))
            readout.scored.append(_score(root, seed, mode, replays[replay], protocol))
    readout.bias = _bias(readout.scored)
    readout.reasons = _gate(readout, protocol)
    readout.decision = "NO-GO" if readout.reasons else "GO"
    return readout


def _identity(identity: dict[str, dict[str, Any]], protocol: Protocol) -> list[str]:
    """Why the builds are not one source's three seeds × five modes of this protocol, complete (module docstring)."""
    out = []
    window = f"{protocol.window_from.isoformat()}/{protocol.window_until.isoformat()}"
    shas = {held.get("git_sha") for held in identity.values()}
    worlds = {held.get("world_sha256") for held in identity.values()}
    if len(shas) != 1 or None in shas:
        out.append(f"not one source: git shas {sorted(str(s) for s in shas)}")
    if len(worlds) != 1:
        out.append("not one world.yaml")
    for name, held in identity.items():
        seed, mode = name.split("/")
        if not held:
            out.append(f"{name}: no build")
            continue
        if str(held.get("master_seed")) != seed or held.get("pilot_mode") != mode:
            out.append(f"{name}: a build of seed {held.get('master_seed')}, mode {held.get('pilot_mode')}")
        if held.get("git_dirty") is not False:
            out.append(f"{name}: built from a dirty checkout")
        if held.get("pilot_window") != window or held.get("last_day_done") != protocol.window_until.isoformat():
            out.append(f"{name}: not done to W's end ({held.get('last_day_done')})")
    for development in protocol.development:
        digests = {identity[f"{development}/{mode}"].get("assignment_digest") for mode in protocol.modes}
        if len(digests) != 1:
            out.append(f"{development}: the modes' assignment maps differ")
    return out


def _score(root: Path, seed: int, mode: str, replay: Frame, protocol: Protocol) -> Scored:
    run = _run(root, seed, mode)
    build = Build(run)
    judged = s5_pilot.judge(run, protocol, build)
    if judged.frame is None or judged.verdict is None:
        return Scored(seed, mode, judged.status, list(judged.reasons), None, None, None, None, None, None, 0)
    codes: dict[Any, str] = dict(build["region"][["region_id", "code"]].itertuples(index=False))
    found = reference(s5_pilot.outcomes(build), replay, codes)
    against = truth(judged.frame, found)
    return Scored(
        seed,
        mode,
        judged.status,
        list(judged.reasons),
        {k: _plain(v) for k, v in asdict(judged.verdict).items()},
        {k: _plain(v) for k, v in asdict(s5_pilot.verdict(against, protocol)).items()},
        _records(judged.frame),
        _records(found),
        {str(k): _plain(v) for k, v in errors(judged.frame, found).items()},
        _records(components(build)),
        judged.later_acquired,
    )


def _bias(scored: Sequence[Scored]) -> dict[str, dict[str, dict[str, float]]]:
    """Per scored mode and region, the normalized errors' mean and RMS over the seeds."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for mode in s5_pilot.SCORED:
        runs = [s.errors for s in scored if s.mode == mode]
        if any(e is None for e in runs):
            continue
        for region in s5_pilot.SKILL:
            values = np.array([e[region] for e in runs if e is not None and e.get(region) is not None], dtype=float)
            if len(values) == len(runs):
                out.setdefault(mode, {})[region] = {
                    "mean": float(values.mean()),
                    "rms": float(np.sqrt((values**2).mean())),
                }
    return out


def _gate(readout: Readout, protocol: Protocol) -> list[str]:
    """Why the development builds are no GO (module docstring); empty for GO."""
    out = []
    bars = protocol.acceptance
    for s in readout.scored:
        name = f"{s.seed}/{s.mode}"
        if s.verdict is None or s.truth is None:
            out.append(f"{name}: {s.status} ({'; '.join(s.reasons)})")
            continue
        if s.mode == "story" and s.status != "PASS":
            out.append(f"{name}: the story is {s.status}, not PASS")
        if s.mode != "story" and not (
            s.status == "REJECTED" and not s.verdict["rank_holds"] and not s.verdict["group_holds"]
        ):
            out.append(f"{name}: the control does not reject both halves ({s.status})")
        if s.truth["status"] != s.status:
            out.append(f"{name}: truth-based {s.truth['status']}, observed {s.status}")
        if s.mode != "story" and s.truth["status"] == "REJECTED" and (s.truth["rank_holds"] or s.truth["group_holds"]):
            out.append(f"{name}: the truth-based control rejects only one half")
    for mode in s5_pilot.SCORED:
        held = readout.bias.get(mode)
        if held is None or len(held) != len(s5_pilot.SKILL):
            out.append(f"{mode}: no normalized errors over the seeds")
            continue
        for region, e in held.items():
            if abs(e["mean"]) > float(bars["bias"]) or e["rms"] > float(bars["rms"]):
                out.append(f"{mode} {region}: mean error {e['mean']:+.3f}, RMS {e['rms']:.3f}")
    return out


# ── the readout ────────────────────────────────────────────────────────────────


def write(readout: Readout, root: Path) -> None:
    (root / f"{READOUT}.json").write_text(
        json.dumps(asdict(readout), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (root / f"{READOUT}.md").write_text(markdown(readout), encoding="utf-8")


def markdown(readout: Readout) -> str:
    """The readout's summary (the JSON holds every number)."""
    lines = [f"# S5 development gate: {readout.decision}", "", f"Builds: `{readout.root}`", ""]
    lines += [f"- {reason}" for reason in readout.reasons] or ["- every condition holds"]
    lines += ["", "## Isolation", "", "| seed | mode | clean | reported | failures |", "|---|---|---|---|---|"]
    for held in readout.isolation:
        reported = ", ".join(f"{k} {n}" for k, n in held["reported"].items())
        lines.append(
            f"| {held['seed']} | {held['mode']} | {held['clean']} | {reported} | {'; '.join(held['failures'])} |"
        )
    if readout.scored:
        lines += [
            "",
            "## Verdicts",
            "",
            "| seed | mode | status | rank | E_H | E_L | E_H/E_L | truth | truth rank | truth E_H/E_L |",
        ]
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for s in readout.scored:
            v, t = s.verdict or {}, s.truth or {}
            lines.append(
                f"| {s.seed} | {s.mode} | {s.status} | {_fmt(v.get('rank'))} | {_fmt(v.get('e_high'))} | "
                f"{_fmt(v.get('e_low'))} | {_fmt(v.get('ratio'))} | {t.get('status', '-')} | {_fmt(t.get('rank'))} | "
                f"{_fmt(t.get('ratio'))} |"
            )
        lines += ["", "## Normalized errors (mean · RMS over the seeds)", ""]
        lines += ["| mode | " + " | ".join(s5_pilot.SKILL) + " |", "|---|" + "---|" * len(s5_pilot.SKILL)]
        for mode, held in readout.bias.items():
            cells = [f"{held[r]['mean']:+.3f} · {held[r]['rms']:.3f}" if r in held else "-" for r in s5_pilot.SKILL]
            lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    return "NULL" if value is None else f"{value:.4f}"


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m hartland_sim.checks.s5_replay ROOT  (ROOT/<seed>/<mode>/ for the development seeds)")
        return 2
    root = Path(args[0])
    readout = develop(root)
    write(readout, root)
    print(markdown(readout), end="")
    return 0 if readout.decision == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
