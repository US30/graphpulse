from kafka import KafkaProducer
import json
import time
import random
import argparse
import logging
import os
from dataclasses import dataclass, asdict
from typing import Generator, Optional
import numpy as np
import uuid
import csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("graphpulse.producer")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Transaction:
    transaction_id: str
    timestamp_unix: float
    amount: float
    card_id: str
    addr_id: str
    features: dict  # 50 numeric features


# ---------------------------------------------------------------------------
# Producer
# ---------------------------------------------------------------------------


class TransactionProducer:
    def __init__(self, bootstrap_servers: str, topic: str) -> None:
        self.topic = topic
        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers.split(","),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            retries=5,
            linger_ms=5,
            compression_type="gzip",
        )
        logger.info(
            "KafkaProducer ready — brokers=%s topic=%s", bootstrap_servers, topic
        )

    def send_transaction(self, tx: Transaction) -> None:
        self._producer.send(self.topic, value=asdict(tx))

    def close(self) -> None:
        self._producer.flush()
        self._producer.close()
        logger.info("KafkaProducer closed.")


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------


class SyntheticTransactionGenerator:
    """Generates statistically plausible synthetic transaction streams."""

    # Approximated from the IEEE-CIS Fraud Detection dataset distribution
    _AMOUNT_MEAN = 150.0
    _AMOUNT_STD = 400.0
    _NUM_CARDS = 50_000
    _NUM_ADDRS = 20_000
    _N_FEATURES = 50

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._card_pool = [
            f"card_{i:06d}" for i in range(self._NUM_CARDS)
        ]
        self._addr_pool = [
            f"addr_{i:05d}" for i in range(self._NUM_ADDRS)
        ]

    def generate(self, fraud_rate: float = 0.01) -> Transaction:
        """Generate one synthetic transaction.

        Fraudulent transactions are distinguished by:
        - Higher amounts (skewed right)
        - Shifted V-feature distributions
        - Small number of known high-risk cards / addresses
        """
        is_fraud = self._rng.random() < fraud_rate

        if is_fraud:
            amount = float(
                abs(self._rng.normal(loc=500.0, scale=600.0))
            )
            # V-features shifted for fraud
            v_features = self._rng.normal(loc=1.5, scale=2.0, size=self._N_FEATURES)
        else:
            amount = float(
                abs(self._rng.normal(loc=self._AMOUNT_MEAN, scale=self._AMOUNT_STD))
            )
            v_features = self._rng.normal(loc=0.0, scale=1.0, size=self._N_FEATURES)

        amount = max(0.01, round(amount, 2))

        features = {f"V{i + 1}": round(float(v_features[i]), 6) for i in range(self._N_FEATURES)}

        card_id = self._rng.choice(self._card_pool[:500] if is_fraud else self._card_pool)
        addr_id = self._rng.choice(self._addr_pool[:200] if is_fraud else self._addr_pool)

        return Transaction(
            transaction_id=str(uuid.uuid4()),
            timestamp_unix=time.time(),
            amount=amount,
            card_id=str(card_id),
            addr_id=str(addr_id),
            features=features,
        )

    def replay_ieee(self, csv_path: str) -> Generator[Transaction, None, None]:
        """Read an IEEE-CIS CSV file and yield Transaction objects in row order.

        Expects the standard IEEE-CIS train_transaction.csv schema with columns:
        TransactionID, isFraud, TransactionDT, TransactionAmt, card1..card6,
        addr1, addr2, V1..V339 (subset of features used).
        """
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    tx_id = str(row.get("TransactionID", str(uuid.uuid4())))
                    amount = float(row.get("TransactionAmt", 0.0))
                    card_id = str(row.get("card1", "unknown"))
                    addr_id = str(row.get("addr1", "unknown"))

                    # Collect V-features (V1..V50 for consistency with synthetic)
                    features: dict = {}
                    for i in range(1, 51):
                        key = f"V{i}"
                        val = row.get(key, "0") or "0"
                        try:
                            features[key] = float(val)
                        except ValueError:
                            features[key] = 0.0

                    yield Transaction(
                        transaction_id=tx_id,
                        timestamp_unix=time.time(),
                        amount=amount,
                        card_id=card_id,
                        addr_id=addr_id,
                        features=features,
                    )
                except Exception as exc:
                    logger.warning("Skipping malformed IEEE row: %s", exc)
                    continue


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GraphPulse Kafka transaction producer."
    )
    parser.add_argument(
        "--tps",
        type=float,
        default=10.0,
        help="Transactions per second to produce (default: 10).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="How long to run in seconds (default: 60). 0 = run forever.",
    )
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=0.01,
        help="Fraction of synthetic transactions labelled as fraudulent (default: 0.01).",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=os.getenv("KAFKA_TOPIC_TRANSACTIONS", "graphpulse.transactions"),
        help="Kafka topic to produce to.",
    )
    parser.add_argument(
        "--bootstrap-servers",
        type=str,
        default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        help="Comma-separated Kafka bootstrap servers.",
    )
    parser.add_argument(
        "--ieee-csv",
        type=str,
        default=None,
        help="Optional path to IEEE-CIS train_transaction.csv to replay real data.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible synthetic data.",
    )
    args = parser.parse_args()

    producer = TransactionProducer(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
    )
    generator = SyntheticTransactionGenerator(seed=args.seed)

    interval = 1.0 / args.tps if args.tps > 0 else 0.0
    deadline = time.time() + args.duration if args.duration > 0 else None

    sent = 0
    errors = 0
    t_stats = time.time()

    logger.info(
        "Starting producer — tps=%.1f duration=%.0fs ieee_csv=%s",
        args.tps,
        args.duration,
        args.ieee_csv or "none",
    )

    try:
        if args.ieee_csv:
            source = generator.replay_ieee(args.ieee_csv)
        else:
            source = None

        while True:
            if deadline and time.time() >= deadline:
                break

            t_loop = time.perf_counter()

            try:
                if source is not None:
                    tx = next(source)
                else:
                    tx = generator.generate(fraud_rate=args.fraud_rate)

                producer.send_transaction(tx)
                sent += 1
            except StopIteration:
                logger.info("IEEE CSV replay complete. %d transactions sent.", sent)
                break
            except Exception as exc:
                errors += 1
                logger.error("Send error: %s", exc)

            # Print throughput stats every 10 seconds
            now = time.time()
            if now - t_stats >= 10.0:
                elapsed = now - t_stats
                logger.info(
                    "Stats — sent=%d errors=%d effective_tps=%.1f",
                    sent,
                    errors,
                    sent / elapsed if elapsed > 0 else 0.0,
                )
                sent = 0
                errors = 0
                t_stats = now

            # Rate limiting
            if interval > 0:
                elapsed_loop = time.perf_counter() - t_loop
                sleep_for = interval - elapsed_loop
                if sleep_for > 0:
                    time.sleep(sleep_for)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        producer.close()
        logger.info("Producer shut down cleanly.")


if __name__ == "__main__":
    main()
