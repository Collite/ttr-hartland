"""The CSV counterpart of the S3 detector in ``data/stories/detectors/s3.sql``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from hartland_sim.checks.stories import Build
    from hartland_sim.config import StoriesConfig

HISTORY_FROM = pd.Timestamp(2024, 1, 1)
SLOT_DAY = pd.Timestamp(2023, 1, 1)
DAY = pd.Timedelta(days=1)
OPENING_BREADTH = pd.Timedelta(days=90)


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


def rates(build: Build) -> pd.DataFrame:
    """The four per-rep rates over the SQL detector's market and date window."""
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
        return pd.DataFrame(columns=["conversion", "acquisition", "churn", "cross_sell"], index=reps, dtype=float)

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

    visits = build.narrow("visit", ("employee_id", "store_id", "customer_id", "kind", "day", "outcome"))
    sales = visits[
        visits["kind"].eq("sales")
        & visits["day"].between(HISTORY_FROM, as_of)
        & visits["store_id"].isin(market["store_id"])
        & visits["customer_id"].isin(retail["customer_id"])
    ]
    conversion = sales["outcome"].eq("order").groupby(sales["employee_id"]).mean().reindex(reps, fill_value=0)
    won = _events(territories, patrons, "since").reindex(reps, fill_value=0)
    # An ended active-day interval can be censored. A true loss starts the next day.
    losses = patrons.loc[patrons["status"].eq("lost") & patrons["until"].lt(as_of), ["area_id", "until"]]
    losses = losses.assign(lost_on=losses["until"] + DAY)
    lost = _events(territories, losses, "lost_on").reindex(reps, fill_value=0)
    added = _events(territories, _added_lines(build, orders, patrons), "ordered_on").reindex(reps, fill_value=0)

    active = active_days.where(active_days.ne(0))
    return pd.DataFrame(
        {
            "conversion": conversion,
            "acquisition": won / potential_days.where(potential_days.ne(0)),
            "churn": lost / active,
            "cross_sell": added / active,
        }
    )


def ranks(observed: pd.DataFrame) -> pd.DataFrame:
    """Competition ranks with missing rates last, then the competition rank of their sum."""
    ordered = observed.drop(columns="churn").rank(ascending=False, method="min", na_option="bottom")
    ordered["churn"] = observed["churn"].rank(method="min", na_option="bottom")
    ordered["score"] = ordered.sum(axis="columns")
    ordered["place"] = ordered["score"].rank(method="min")
    return ordered


def s3(build: Build, stories: StoriesConfig) -> float:
    """Count story pins in their required placement bands, including when the build is a null control."""
    places = ranks(rates(build))["place"]
    if places.empty:
        return 0.0
    assignments = build.narrow("assignment", ("employee_id", "area_id", "kind", "valid_from", "valid_to"))
    slots = assignments[
        assignments["kind"].eq("rep_area")
        & assignments["valid_from"].le(SLOT_DAY)
        & assignments["valid_to"].ge(SLOT_DAY)
    ].merge(build.narrow("area", ("area_id", "code")), on="area_id")
    profiles = slots["code"].map({pin.slot: pin.profile for pin in stories.pins.reps})
    placement = slots["employee_id"].map(places)
    stars = profiles.eq("star") & placement.le(5)
    weak = profiles.eq("weak") & placement.gt(len(places) - 5)
    return float((stars | weak).sum())
