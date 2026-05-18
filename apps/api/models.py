from sqlalchemy import Column, String, Float, Boolean, DateTime, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
import uuid
import datetime

Base = declarative_base()


class TransactionScore(Base):
    """Persisted record of every fraud scoring decision."""

    __tablename__ = "transaction_scores"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    transaction_id = Column(String(64), nullable=False, index=True)
    fraud_score = Column(Float, nullable=False)
    is_fraud = Column(Boolean, nullable=False)
    model = Column(String(32), nullable=False)
    # Serialised JSON of the raw feature dict — nullable to keep the row slim
    # for high-volume scenarios where features are already stored upstream.
    features_json = Column(Text, nullable=True)
    # Top-5 SHAP feature importances as JSON {"feature": shap_value, ...}
    shap_json = Column(Text, nullable=True)
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.datetime.utcnow,
    )

    def __repr__(self) -> str:
        return (
            f"<TransactionScore id={self.id} "
            f"tx={self.transaction_id} "
            f"score={self.fraud_score:.4f} "
            f"fraud={self.is_fraud}>"
        )


class DriftAlert(Base):
    """Records drift signals emitted by the monitoring layer."""

    __tablename__ = "drift_alerts"

    ALERT_TYPES = ("score_drift", "error_drift", "concept_drift")

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    # One of: score_drift | error_drift | concept_drift
    alert_type = Column(String(64), nullable=False, index=True)
    triggered_at = Column(
        DateTime,
        nullable=False,
        default=datetime.datetime.utcnow,
    )
    # Arbitrary JSON payload: window stats, detector state, feature drift scores, etc.
    details_json = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<DriftAlert id={self.id} "
            f"type={self.alert_type} "
            f"at={self.triggered_at.isoformat()}>"
        )
