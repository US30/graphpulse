from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, FileSource, ValueType
from feast.types import Float32, Int64, String
from feast import Field
from feast import FeatureStore


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

card_entity = Entity(
    name="card_id",
    join_keys=["card_id"],
    value_type=ValueType.STRING,
    description="Composite card identifier (card1_card2).",
)

address_entity = Entity(
    name="addr_id",
    join_keys=["addr_id"],
    value_type=ValueType.STRING,
    description="Composite address identifier (addr1_addr2).",
)

transaction_entity = Entity(
    name="transaction_id",
    join_keys=["TransactionID"],
    value_type=ValueType.INT64,
    description="Unique transaction identifier from IEEE-CIS dataset.",
)


# ---------------------------------------------------------------------------
# Feature Sources
# ---------------------------------------------------------------------------

card_features_source = FileSource(
    path="data/features/card_features.parquet",
    timestamp_field="event_timestamp",
    description="Pre-materialised card-level aggregated features.",
)

transaction_features_source = FileSource(
    path="data/features/transactions.parquet",
    timestamp_field="event_timestamp",
    description="Pre-materialised transaction-level features.",
)


# ---------------------------------------------------------------------------
# Feature Views
# ---------------------------------------------------------------------------

card_features_fv = FeatureView(
    name="card_features",
    entities=[card_entity],
    ttl=timedelta(days=7),
    source=card_features_source,
    schema=[
        Field(name="tx_count_24h", dtype=Float32),
        Field(name="fraud_rate_7d", dtype=Float32),
        Field(name="avg_amount_7d", dtype=Float32),
        Field(name="max_amount_30d", dtype=Float32),
        Field(name="unique_addr_7d", dtype=Int64),
        Field(name="tx_velocity_1h", dtype=Float32),
    ],
    description="Card-level rolling aggregation features (24h, 7d, 30d windows).",
)

transaction_features_fv = FeatureView(
    name="transaction_features",
    entities=[transaction_entity],
    ttl=timedelta(days=1),
    source=transaction_features_source,
    schema=[
        Field(name="TransactionAmt_log", dtype=Float32),
        Field(name="hour_of_day", dtype=Int64),
        Field(name="day_of_week", dtype=Int64),
        Field(name="P_emaildomain", dtype=String),
        Field(name="R_emaildomain", dtype=String),
        Field(name="card_country", dtype=String),
        Field(name="dist1", dtype=Float32),
        Field(name="dist2", dtype=Float32),
    ],
    description="Transaction-level features including time signals and anonymised fields.",
)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def get_feature_store(repo_path: str = ".") -> FeatureStore:
    """Return a Feast FeatureStore pointing at the given repo directory.

    Parameters
    ----------
    repo_path : str
        Path to the Feast feature repository (directory containing feature_store.yaml).

    Returns
    -------
    FeatureStore
    """
    return FeatureStore(repo_path=repo_path)


def materialize_features(store: FeatureStore, start: str, end: str) -> None:
    """Materialise all registered feature views between two ISO-8601 date strings.

    Parameters
    ----------
    store : FeatureStore
        An initialised Feast FeatureStore instance.
    start : str
        Start datetime in ISO-8601 format, e.g. ``"2020-01-01T00:00:00"``.
    end : str
        End datetime in ISO-8601 format, e.g. ``"2020-12-31T23:59:59"``.

    Example
    -------
    >>> store = get_feature_store(".")
    >>> materialize_features(store, "2020-01-01", "2020-12-31")
    """
    from datetime import datetime

    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    store.materialize(start_date=start_dt, end_date=end_dt)
