"""
download_ieee.py
----------------
Download the IEEE-CIS Fraud Detection dataset from Kaggle.

Usage:
    python scripts/download_ieee.py --output-dir data/raw/ieee_cis

Requirements:
    pip install kaggle
    Set KAGGLE_USERNAME and KAGGLE_KEY environment variables,
    OR place ~/.kaggle/kaggle.json with your credentials.
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

COMPETITION = "ieee-fraud-detection"
REQUIRED_FILES = [
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
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


def download_ieee(output_dir: Path = Path("data/raw/ieee_cis")) -> None:
    """
    Download, unzip, and verify the IEEE-CIS Fraud Detection dataset.

    Parameters
    ----------
    output_dir : Path
        Directory where the dataset files will be stored.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Checking Kaggle credentials...")
    _check_kaggle_credentials()

    log.info("Downloading IEEE-CIS dataset (this may take several minutes — ~370 MB)...")
    cmd = [
        "kaggle",
        "competitions",
        "download",
        "-c", COMPETITION,
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
            "Download failed. Make sure you have accepted the competition rules at:\n"
            f"  https://www.kaggle.com/competitions/{COMPETITION}/rules"
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

    # Verify required files
    log.info("Verifying required files...")
    missing = []
    for fname in REQUIRED_FILES:
        fpath = output_dir / fname
        if not fpath.exists():
            missing.append(fname)
        else:
            size_mb = fpath.stat().st_size / 1_048_576
            log.info("  %-35s %8.1f MB", fname, size_mb)

    if missing:
        raise FileNotFoundError(
            f"Download completed but the following required files are missing:\n"
            + "\n".join(f"  - {f}" for f in missing)
            + f"\n\nCheck {output_dir} for partial downloads."
        )

    log.info("Dataset ready at: %s", output_dir.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the IEEE-CIS Fraud Detection dataset via Kaggle API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/ieee_cis"),
        help="Destination directory for the dataset.",
    )
    args = parser.parse_args()
    download_ieee(args.output_dir)


if __name__ == "__main__":
    main()
