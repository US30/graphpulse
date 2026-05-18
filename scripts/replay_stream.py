"""
Replay IEEE-CIS transaction CSV into Redpanda (Kafka API) at configurable TPS.

Usage:
    python scripts/replay_stream.py \
        --csv data/raw/ieee_cis/train_transaction.csv \
        --tps 500 \
        --duration 120
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
import csv
from pathlib import Path

logger = logging.getLogger(__name__)


def _parse_row(row: dict) -> dict:
    """Convert a CSV row dict to a scoring-API-compatible transaction dict."""
    features: dict = {}
    for i in range(1, 51):
        key = f"V{i}"
        val = row.get(key, "0") or "0"
        try:
            features[key] = float(val)
        except ValueError:
            features[key] = 0.0

    return {
        "transaction_id": str(row.get("TransactionID", uuid.uuid4())),
        "timestamp_unix": time.time(),
        "amount": float(row.get("TransactionAmt", 0.0) or 0.0),
        "card_id": str(row.get("card1", "unknown")),
        "addr_id": str(row.get("addr1", "unknown")),
        "is_fraud_ground_truth": int(row.get("isFraud", 0) or 0),
        "features": features,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Replay IEEE-CIS CSV → Kafka/Redpanda topic.")
    parser.add_argument(
        "--csv",
        default="data/raw/ieee_cis/train_transaction.csv",
        help="Path to IEEE-CIS train_transaction.csv",
    )
    parser.add_argument(
        "--tps", type=float, default=100.0, help="Target transactions per second (default: 100)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0,
        help="How long to replay in seconds (0 = replay entire file once)",
    )
    parser.add_argument(
        "--topic",
        default=os.getenv("KAFKA_TOPIC_TRANSACTIONS", "graphpulse.transactions"),
        help="Kafka topic to produce to",
    )
    parser.add_argument(
        "--bootstrap-servers",
        default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        help="Comma-separated Kafka bootstrap servers",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the file until --duration is exhausted",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    try:
        from kafka import KafkaProducer

        producer = KafkaProducer(
            bootstrap_servers=args.bootstrap_servers.split(","),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks=1,
            linger_ms=2,
            compression_type="gzip",
        )
        logger.info("Kafka producer connected to %s", args.bootstrap_servers)
    except Exception as exc:
        logger.error("Failed to create Kafka producer: %s", exc)
        raise

    interval = 1.0 / args.tps if args.tps > 0 else 0.0
    deadline = time.time() + args.duration if args.duration > 0 else None

    sent = 0
    errors = 0
    t_stats = time.time()
    loop_count = 0

    logger.info(
        "Replaying %s → topic=%s tps=%.0f loop=%s",
        csv_path.name, args.topic, args.tps, args.loop,
    )

    try:
        while True:
            loop_count += 1
            with open(csv_path, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if deadline and time.time() >= deadline:
                        break

                    t_loop = time.perf_counter()

                    try:
                        tx = _parse_row(row)
                        producer.send(args.topic, value=tx)
                        sent += 1
                    except Exception as exc:
                        errors += 1
                        logger.warning("Send error: %s", exc)

                    now = time.time()
                    if now - t_stats >= 5.0:
                        logger.info(
                            "Loop %d | sent=%d errors=%d effective_tps=%.0f",
                            loop_count, sent, errors,
                            sent / (now - t_stats) if (now - t_stats) > 0 else 0,
                        )
                        sent = 0
                        errors = 0
                        t_stats = now

                    if interval > 0:
                        elapsed = time.perf_counter() - t_loop
                        sleep_for = interval - elapsed
                        if sleep_for > 0:
                            time.sleep(sleep_for)
                else:
                    if not args.loop:
                        break
                    continue
                break  # deadline hit inside inner loop

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        producer.flush()
        producer.close()
        logger.info("Replay complete. Total sent=%d errors=%d", sent, errors)


if __name__ == "__main__":
    main()
