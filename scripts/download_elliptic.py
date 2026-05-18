"""
download_elliptic.py
--------------------
Download the Elliptic Bitcoin Dataset from Kaggle.

Usage:
    python scripts/download_elliptic.py --output-dir data/raw/elliptic

Requirements:
    pip install kaggle
    Set KAGGLE_USERNAME and KAGGLE_KEY environment variables,
    OR place ~/.kaggle/kaggle.json with your credentials.

Dataset: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
Citation: Weber et al. (2019). "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics."
"""

import subprocess
import os
import zipfile
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

DATASET_SLUG = "ellipticco/elliptic-data-set"
REQUIRED_FILES = [
    "elliptic_txs_features.csv",
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv",
]


def _check_kaggle_credentials() -> None:
    """Raise RuntimeError with setup instructions if Kaggle credentials are missing."""
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

    if (username and key) or kaggle_json.exists():
        return

    raise RuntimeError(
        "Kaggle credentials not found.\n\n"
        "Option 1 — Environment variables (recommended for CI):\n"
        "  export KAGGLE_USERNAME=<your-username>\n"
        "  export KAGGLE_KEY=<your-api-key>\n\n"
        "Option 2 — Credential file:\n"
        "  mkdir -p ~/.kaggle && chmod 700 ~/.kaggle\n"
        '  echo \'{"username":"<your-username>","key":"<your-api-key>"}\' > ~/.kaggle/kaggle.json\n'
        "  chmod 600 ~/.kaggle/kaggle.json\n\n"
        "Get your API key at: https://www.kaggle.com/settings → API → Create New Token"
    )


def download_elliptic(output_dir: Path = Path("data/raw/elliptic")) -> None:
    """
    Download, unzip, and verify the Elliptic Bitcoin Dataset.

    Parameters
    ----------
    output_dir : Path
        Directory where the dataset files will be stored.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Checking Kaggle credentials...")
    _check_kaggle_credentials()

    log.info("Downloading Elliptic dataset (this may take a few minutes — ~70 MB)...")
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d", DATASET_SLUG,
        "-p", str(output_dir),
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        log.info(result.stdout.strip())
    except subprocess.CalledProcessError as exc:
        log.error("kaggle CLI returned non-zero exit code: %s", exc.returncode)
        log.error(exc.stderr)
        raise RuntimeError(
            "Download failed. Ensure your Kaggle credentials are valid and you have\n"
            f"internet access. Dataset: https://www.kaggle.com/datasets/{DATASET_SLUG}"
        ) from exc

    # Unzip all archives in the output dir
    zip_files = list(output_dir.glob("*.zip"))
    if not zip_files:
        log.warning("No zip files found in %s — skipping extraction.", output_dir)
    for zf_path in zip_files:
        log.info("Extracting %s ...", zf_path.name)
        with zipfile.ZipFile(zf_path, "r") as zf:
            zf.extractall(output_dir)
        zf_path.unlink()
        log.info("Deleted archive %s", zf_path.name)

    # Some versions nest inside a subdirectory — flatten if needed
    nested_dir = output_dir / "elliptic_bitcoin_dataset"
    if nested_dir.exists():
        log.info("Flattening nested directory %s ...", nested_dir.name)
        for child in nested_dir.iterdir():
            child.rename(output_dir / child.name)
        nested_dir.rmdir()

    # Verify required files
    log.info("Verifying required files...")
    missing = []
    for fname in REQUIRED_FILES:
        fpath = output_dir / fname
        if not fpath.exists():
            missing.append(fname)
        else:
            size_mb = fpath.stat().st_size / 1_048_576
            log.info("  %-40s %8.1f MB", fname, size_mb)

    if missing:
        raise FileNotFoundError(
            f"Download completed but the following required files are missing:\n"
            + "\n".join(f"  - {f}" for f in missing)
            + f"\n\nCheck {output_dir} for partial downloads or unexpected directory layout."
        )

    log.info("Dataset ready at: %s", output_dir.resolve())
    log.info(
        "Dataset description:\n"
        "  elliptic_txs_features.csv  — 203k nodes, 166 features (anonymous)\n"
        "  elliptic_txs_classes.csv   — labels: 1=illicit, 2=licit, unknown\n"
        "  elliptic_txs_edgelist.csv  — 234k directed edges (payment flows)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Elliptic Bitcoin Dataset via Kaggle API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/elliptic"),
        help="Destination directory for the dataset.",
    )
    args = parser.parse_args()
    download_elliptic(args.output_dir)


if __name__ == "__main__":
    main()
