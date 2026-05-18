from kafka import KafkaConsumer, KafkaProducer
import json
import httpx
import asyncio
import logging
import os
from datetime import datetime
from typing import Optional
import uuid
import psycopg2
import psycopg2.extras
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("graphpulse.consumer")


# ---------------------------------------------------------------------------
# Consumer
# ---------------------------------------------------------------------------


class FraudConsumer:
    """Kafka consumer that scores transactions via the FastAPI service and
    writes results to PostgreSQL + a downstream scores topic."""

    def __init__(
        self,
        bootstrap_servers: str,
        input_topic: str,
        output_topic: str,
        api_url: str,
        db_dsn: Optional[str] = None,
    ) -> None:
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.api_url = api_url.rstrip("/")

        servers = bootstrap_servers.split(",")

        self._consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=servers,
            group_id="graphpulse-consumer",
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            auto_commit_interval_ms=5000,
            max_poll_records=100,
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
        )
        logger.info("KafkaConsumer ready — topic=%s", input_topic)

        self._producer = KafkaProducer(
            bootstrap_servers=servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            retries=3,
            linger_ms=5,
            compression_type="gzip",
        )
        logger.info("KafkaProducer ready — topic=%s", output_topic)

        self._http = httpx.Client(base_url=self.api_url, timeout=5.0)

        # PostgreSQL — optional; degrades gracefully if unavailable
        self._db: Optional[psycopg2.extensions.connection] = None
        dsn = db_dsn or os.getenv("DATABASE_URL")
        if dsn:
            try:
                self._db = psycopg2.connect(dsn)
                self._db.autocommit = False
                self._ensure_table()
                logger.info("PostgreSQL connected.")
            except Exception as exc:
                logger.warning("PostgreSQL unavailable (%s) — DB writes disabled.", exc)
                self._db = None

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _ensure_table(self) -> None:
        """Create the transaction_scores table if it doesn't exist yet."""
        ddl = """
        CREATE TABLE IF NOT EXISTS transaction_scores (
            id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            transaction_id VARCHAR(64) NOT NULL,
            fraud_score  DOUBLE PRECISION NOT NULL,
            is_fraud     BOOLEAN NOT NULL,
            model        VARCHAR(32) NOT NULL,
            features_json TEXT,
            shap_json    TEXT,
            created_at   TIMESTAMP NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_ts_transaction_id
            ON transaction_scores (transaction_id);
        """
        with self._db.cursor() as cur:
            cur.execute(ddl)
        self._db.commit()

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process_message(self, msg: dict) -> dict:
        """Call POST /score on the API service and return the score response."""
        payload = {
            "transaction_id": msg.get("transaction_id", str(uuid.uuid4())),
            "features": msg.get("features", {}),
            "timestamp_unix": msg.get("timestamp_unix", datetime.utcnow().timestamp()),
            "model": msg.get("model", "lgbm"),
        }

        try:
            resp = self._http.post("/score", json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "API HTTP error %s for tx=%s: %s",
                exc.response.status_code,
                payload["transaction_id"],
                exc.response.text,
            )
            raise
        except httpx.RequestError as exc:
            logger.error(
                "API request error for tx=%s: %s", payload["transaction_id"], exc
            )
            raise

    def write_score(self, score: dict) -> None:
        """Insert score into PostgreSQL and publish to the output Kafka topic."""
        # Publish to downstream topic regardless of DB availability
        try:
            self._producer.send(self.output_topic, value=score)
        except Exception as exc:
            logger.error("Failed to publish score to Kafka: %s", exc)

        # Write to PostgreSQL
        if self._db is None:
            return

        insert_sql = """
        INSERT INTO transaction_scores
            (transaction_id, fraud_score, is_fraud, model, created_at)
        VALUES (%s, %s, %s, %s, NOW())
        ON CONFLICT DO NOTHING;
        """
        try:
            with self._db.cursor() as cur:
                cur.execute(
                    insert_sql,
                    (
                        score.get("transaction_id"),
                        score.get("fraud_score"),
                        score.get("is_fraud"),
                        score.get("model"),
                    ),
                )
            self._db.commit()
        except Exception as exc:
            logger.error("DB write failed: %s", exc)
            try:
                self._db.rollback()
            except Exception:
                pass

    def run(self, max_messages: Optional[int] = None) -> None:
        """Poll the Kafka consumer, score each message, write results.

        Args:
            max_messages: Stop after processing this many messages.
                          None means run until interrupted.
        """
        processed = 0
        errors = 0
        t_start = datetime.utcnow()

        logger.info(
            "Consumer started — input=%s output=%s api=%s max=%s",
            self.input_topic,
            self.output_topic,
            self.api_url,
            max_messages or "unlimited",
        )

        try:
            for kafka_msg in self._consumer:
                msg: dict = kafka_msg.value

                try:
                    score = self.process_message(msg)
                    self.write_score(score)
                    processed += 1

                    if score.get("is_fraud"):
                        logger.warning(
                            "FRAUD detected — tx=%s score=%.4f model=%s",
                            score.get("transaction_id"),
                            score.get("fraud_score", 0.0),
                            score.get("model"),
                        )
                except Exception as exc:
                    errors += 1
                    logger.error("Failed to process message: %s", exc)

                # Throughput logging every 100 messages
                if processed % 100 == 0 and processed > 0:
                    elapsed = (datetime.utcnow() - t_start).total_seconds()
                    tps = processed / elapsed if elapsed > 0 else 0.0
                    logger.info(
                        "Throughput — processed=%d errors=%d elapsed=%.1fs tps=%.1f",
                        processed,
                        errors,
                        elapsed,
                        tps,
                    )

                if max_messages is not None and processed >= max_messages:
                    logger.info("Reached max_messages=%d — stopping.", max_messages)
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self._shutdown()

        elapsed_total = (datetime.utcnow() - t_start).total_seconds()
        logger.info(
            "Consumer stopped — processed=%d errors=%d total_elapsed=%.1fs",
            processed,
            errors,
            elapsed_total,
        )

    def _shutdown(self) -> None:
        logger.info("Shutting down consumer resources...")
        try:
            self._consumer.close()
        except Exception:
            pass
        try:
            self._producer.flush()
            self._producer.close()
        except Exception:
            pass
        try:
            self._http.close()
        except Exception:
            pass
        if self._db:
            try:
                self._db.close()
            except Exception:
                pass
        logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GraphPulse Kafka fraud consumer."
    )
    parser.add_argument(
        "--bootstrap-servers",
        type=str,
        default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        help="Comma-separated Kafka bootstrap servers.",
    )
    parser.add_argument(
        "--input-topic",
        type=str,
        default=os.getenv("KAFKA_TOPIC_TRANSACTIONS", "graphpulse.transactions"),
        help="Kafka topic to consume raw transactions from.",
    )
    parser.add_argument(
        "--output-topic",
        type=str,
        default=os.getenv("KAFKA_TOPIC_SCORES", "graphpulse.scores"),
        help="Kafka topic to publish fraud scores to.",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.getenv("SCORING_API_URL", "http://localhost:8000"),
        help="Base URL of the GraphPulse scoring API.",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Stop after this many messages (useful for smoke tests).",
    )
    parser.add_argument(
        "--db-dsn",
        type=str,
        default=None,
        help="PostgreSQL DSN (falls back to DATABASE_URL env var).",
    )
    args = parser.parse_args()

    consumer = FraudConsumer(
        bootstrap_servers=args.bootstrap_servers,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        api_url=args.api_url,
        db_dsn=args.db_dsn,
    )
    consumer.run(max_messages=args.max_messages)


if __name__ == "__main__":
    main()
