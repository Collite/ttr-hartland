"""The CSV counterpart of the S3 detector in ``data/stories/detectors/s3.sql``."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, cast

import pandas as pd

if TYPE_CHECKING:
    from hartland_sim.checks.stories import Build
    from hartland_sim.config import StoriesConfig

HISTORY_FROM = pd.Timestamp(2024, 1, 1)
SLOT_DAY = pd.Timestamp(2023, 1, 1)
DAY = pd.Timedelta(days=1)
OPENING_BREADTH = pd.Timedelta(days=90)
MEASURES = {
    "conversion": ("converting_visits", "sales_visits"),
    "acquisition": ("wins", "potential_days"),
    "retention": ("losses", "active_days"),
    "cross_sell": ("first_paid_lines", "active_days"),
}


def _days(duty: pd.DataFrame, in_area: pd.DataFrame, first: str, last: str) -> pd.Series:
    span = duty.merge(in_area, on="area_id")
    start = span[first].where(span[first] > span["from_day"], span["from_day"])
    end = span[last].where(span[last] < span["to_day"], span["to_day"])
    return ((end - start).dt.days + 1).clip(lower=0).groupby(span["employee_id"]).sum()


def _events(duty: pd.DataFrame, in_area: pd.DataFrame, on: str) -> pd.Series:
    events = duty.merge(in_area, on="area_id")
    return events[events[on].between(events["from_day"], events["to_day"])].groupby("employee_id").size()


def _added_lines(build: Build, orders: pd.DataFrame, patrons: pd.DataFrame) -> pd.DataFrame:
    """A first non-free line after the opening period, attributed on the first order day."""
    small = orders[orders["channel"].eq("small_retail") & orders["customer_id"].isin(patrons["customer_id"])]
    lines = build.narrow("sales_order_line", ("sales_order_id", "sku_id", "is_free_goods"))
    paid = lines.loc[lines["is_free_goods"].eq(False), ["sales_order_id", "sku_id"]]
    sku = build.narrow("sku", ("sku_id", "product_id"))
    products = build.narrow("product", ("product_id", "product_line_id"))
    sales = small.merge(paid, on="sales_order_id").merge(sku.merge(products, on="product_id"), on="sku_id")
    first = sales.groupby(["customer_id", "product_line_id"], as_index=False)["ordered_on"].min()
    first = first.merge(patrons[["customer_id", "area_id", "since"]], on="customer_id")
    opening = first["since"].where(first["since"].ge(HISTORY_FROM), HISTORY_FROM)
    added: pd.DataFrame = first.loc[first["ordered_on"].gt(opening + OPENING_BREADTH)]
    return added


def counts(build: Build, *, reconcile: bool = False) -> pd.DataFrame:
    """Numerators and exposure over the SQL detector's market and date window."""
    assignments = build.narrow("assignment", ("employee_id", "area_id", "kind", "valid_from", "valid_to"))
    orders = build.narrow("sales_order", ("sales_order_id", "customer_id", "ordered_on", "channel"))
    as_of = orders["ordered_on"].max()
    duty = assignments.loc[
        assignments["kind"].eq("rep_area")
        & assignments["valid_to"].ge(HISTORY_FROM)
        & assignments["valid_from"].le(as_of)
    ].copy()
    duty["from_day"] = duty["valid_from"].where(duty["valid_from"].ge(HISTORY_FROM), HISTORY_FROM)
    duty["to_day"] = duty["valid_to"].where(duty["valid_to"].le(as_of), as_of)
    reps = pd.Index(duty["employee_id"].unique(), name="employee_id")
    if duty.empty:
        columns = list(dict.fromkeys(column for pair in MEASURES.values() for column in pair))
        if reconcile:
            columns.extend(("linked_orders", "link_mismatches"))
        return pd.DataFrame(columns=columns, index=reps, dtype="int64")

    stores = build.narrow("store", ("store_id", "area_id", "chain_id", "store_class", "opened_on", "closed_on"))
    chains = build.narrow("chain", ("chain_id", "is_key_account"))
    key_accounts = chains.loc[chains["is_key_account"].eq(True), "chain_id"]
    market = stores[stores["store_class"].ne("horeca") & ~stores["chain_id"].isin(key_accounts)]
    customers = build.narrow("customer", ("customer_id", "store_id", "channel", "since", "until", "status"))
    retail = customers[customers["channel"].eq("small_retail")]
    patrons = retail[retail["since"].notna()].merge(market[["store_id", "area_id"]], on="store_id")
    territories = duty[duty["area_id"].notna()]
    store_days = _days(territories, market, "opened_on", "closed_on").reindex(reps)
    active_days = _days(territories, patrons, "since", "until").reindex(reps)
    potential_days = store_days - active_days.fillna(0)

    visit_columns = ("employee_id", "store_id", "customer_id", "kind", "day", "outcome")
    visits = build.narrow("visit", (*visit_columns, "visit_id") if reconcile else visit_columns)
    sales = visits[
        visits["kind"].eq("sales")
        & visits["day"].between(HISTORY_FROM, as_of)
        & visits["store_id"].isin(market["store_id"])
        & visits["customer_id"].isin(retail["customer_id"])
    ]
    converting = sales["outcome"].eq("order")
    visits_per_rep = sales.groupby("employee_id").size().reindex(reps, fill_value=0)
    conversions = converting.groupby(sales["employee_id"]).sum().reindex(reps, fill_value=0)
    won = _events(territories, patrons, "since").reindex(reps, fill_value=0)
    # An ended active-day interval can be censored. A true loss starts the next day.
    losses = patrons.loc[patrons["status"].eq("lost") & patrons["until"].lt(as_of), ["area_id", "until"]]
    losses = losses.assign(lost_on=losses["until"] + DAY)
    lost = _events(territories, losses, "lost_on").reindex(reps, fill_value=0)
    added = _events(territories, _added_lines(build, orders, patrons), "ordered_on").reindex(reps, fill_value=0)

    observed = pd.DataFrame(
        {
            "sales_visits": visits_per_rep,
            "converting_visits": conversions,
            "potential_days": potential_days.fillna(0),
            "wins": won,
            "active_days": active_days.fillna(0),
            "losses": lost,
            "first_paid_lines": added,
        },
        dtype="int64",
    )
    if reconcile:
        orders_per_visit = build.narrow("sales_order", ("visit_id",)).groupby("visit_id").size()
        linked = sales["visit_id"].map(orders_per_visit).fillna(0).astype("int64")
        observed["linked_orders"] = linked.groupby(sales["employee_id"]).sum().reindex(reps, fill_value=0)
        mismatch = linked.ne(converting.astype("int64"))
        observed["link_mismatches"] = mismatch.groupby(sales["employee_id"]).sum().reindex(reps, fill_value=0)
    return observed


def exact_rates(observed: pd.DataFrame) -> pd.DataFrame:
    """Signed fractions for ranking, with zero or missing exposure undefined."""
    return pd.DataFrame(
        {
            measure: [
                Fraction(int(n) * (-1 if measure == "retention" else 1), int(d)) if d > 0 else None
                for n, d in zip(observed[numerator], observed[exposure], strict=True)
            ]
            for measure, (numerator, exposure) in MEASURES.items()
        },
        index=observed.index,
    )


def rates(build: Build) -> pd.DataFrame:
    """Historical rates, including the original convention for reps without sales visits."""
    values = exact_rates(build.joined("s3 counts", counts)).astype(float)
    values["conversion"] = values["conversion"].fillna(0)
    return values.rename(columns={"retention": "churn"}).assign(churn=-values["retention"])


def ranks(observed: pd.DataFrame) -> pd.DataFrame:
    """Competition ranks with missing rates last, then the competition rank of their sum."""
    ordered = observed.drop(columns="churn").rank(ascending=False, method="min", na_option="bottom")
    ordered["churn"] = observed["churn"].rank(method="min", na_option="bottom")
    ordered["score"] = ordered.sum(axis="columns")
    ordered["place"] = ordered["score"].rank(method="min")
    return ordered


def slots(build: Build) -> pd.DataFrame:
    """Warm-up identity by employee, independent of the employee's subsequent territory."""
    assignments = build.narrow("assignment", ("employee_id", "area_id", "kind", "valid_from", "valid_to"))
    return assignments[
        assignments["kind"].eq("rep_area")
        & assignments["valid_from"].le(SLOT_DAY)
        & assignments["valid_to"].ge(SLOT_DAY)
    ].merge(build.narrow("area", ("area_id", "code")), on="area_id")


def s3(build: Build, stories: StoriesConfig) -> float:
    """Count story pins in their required placement bands, including when the build is a null control."""
    places = ranks(rates(build))["place"]
    if places.empty:
        return 0.0
    warmup = slots(build)
    profiles = warmup["code"].map({pin.slot: pin.profile for pin in stories.pins.reps})
    placement = warmup["employee_id"].map(places)
    stars = profiles.eq("star") & placement.le(5)
    weak = profiles.eq("weak") & placement.gt(len(places) - 5)
    return float((stars | weak).sum())


def business_report(build: Build, stories: StoriesConfig) -> str:
    """Observed counts, exposure, signed rates and competition ranks, without the hidden answer key."""
    observed = build.joined("s3 counts", counts)
    warmup = slots(build).set_index("employee_id")["code"]
    if observed.empty or not warmup.index.is_unique:
        return ""
    measures = exact_rates(observed)
    ordered = measures.rank(method="min", ascending=False)
    titles = ("Konverze", "Akvizice", "Udržení", "Šíře sortimentu")
    rows = [
        "### Pozorované výsledky obchodních zástupců",
        "",
        "Každá buňka uvádí počet, základ, míru a pořadí. Míra udržení je záporná míra ztrát.",
        "",
        "| Zástupce | Oblast na začátku | " + " | ".join(titles) + " |",
        "|---|---|" + "---:|" * len(titles),
    ]
    for employee in observed.sort_index().index:
        cells = []
        for measure, (numerator, exposure) in MEASURES.items():
            n = cast(int, observed.at[employee, numerator])
            d = cast(int, observed.at[employee, exposure])
            rate = cast(Fraction | None, measures.at[employee, measure])
            rank = cast(float, ordered.at[employee, measure])
            display = "—" if rate is None else f"{float(rate):.6f} · {int(rank)}"
            cells.append(f"{n} / {d} · {display}")
        rows.append(f"| {employee} | {warmup.get(employee, '—')} | " + " | ".join(cells) + " |")
    historical = ranks(rates(build))["place"].to_dict()
    names = {pin.slot for pin in stories.pins.reps if pin.profile in {"star", "weak"}}
    places = [
        f"{code} {int(historical[employee])}."
        for employee, code in warmup.items()
        if code in names and employee in historical
    ]
    if places:
        rows.extend(("", "Původní souhrnné pořadí · " + ", ".join(places)))
    return "\n".join(rows) + "\n"
