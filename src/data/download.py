"""Dataset downloader for Superviz26-SQL from Zenodo.

Authoritative Zenodo DOI: 10.5281/zenodo.19627322
Repository: https://github.com/gquetel/sqlia-dataset
"""

import logging
import os
import sys
import time
from pathlib import Path
import urllib.request
import urllib.error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ZENODO_RECORD_ID = "19627322"
BASE_ZENODO_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files"

# Canonical file sizes from Zenodo API
EXPECTED_SIZES = {
    "d-d.csv": 227017688,
    "abc-d.csv": 233392756,
    "a-a.csv": 314230801,
    "b-b.csv": 271108203,
    "c-c.csv": 338373492,
}


def download_zenodo_file(filename: str, output_dir: Path, force: bool = False) -> Path:
    """Download a single CSV file from Zenodo with progress tracking.

    Args:
        filename: Name of the file on Zenodo (e.g. 'd-d.csv').
        output_dir: Local destination directory.
        force: If True, redownload even if file exists with matching size.

    Returns:
        Path to the downloaded file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / filename

    expected_size = EXPECTED_SIZES.get(filename)
    if target_path.exists() and not force:
        actual_size = target_path.stat().st_size
        if expected_size is None or actual_size == expected_size:
            logger.info(f"File {filename} already exists ({actual_size:,} bytes). Skipping download.")
            return target_path
        else:
            logger.warning(
                f"File {filename} exists but size mismatch ({actual_size:,} vs expected {expected_size:,}). Redownloading."
            )

    url = f"{BASE_ZENODO_URL}/{filename}/content"
    logger.info(f"Initiating download of {filename} from {url}...")
    temp_path = target_path.with_suffix(".tmp")

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SQLi-Research/1.0"}
    req = urllib.request.Request(url, headers=headers)

    start_time = time.time()
    last_log_time = start_time
    downloaded_bytes = 0

    try:
        with urllib.request.urlopen(req) as response, open(temp_path, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", expected_size or 0))
            chunk_size = 1024 * 1024  # 1 MB chunk

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded_bytes += len(chunk)

                current_time = time.time()
                if current_time - last_log_time >= 5.0 or (total_size and downloaded_bytes == total_size):
                    elapsed = current_time - start_time
                    speed_mb = (downloaded_bytes / (1024 * 1024)) / max(elapsed, 0.001)
                    pct = (downloaded_bytes / total_size * 100) if total_size else 0.0
                    logger.info(
                        f"Downloading {filename}: {downloaded_bytes / (1024*1024):.1f} MB / "
                        f"{total_size / (1024*1024):.1f} MB ({pct:.1f}%) at {speed_mb:.2f} MB/s"
                    )
                    last_log_time = current_time

        if temp_path.exists():
            temp_path.replace(target_path)
            logger.info(f"Download complete: {target_path} ({target_path.stat().st_size:,} bytes).")
            return target_path
        else:
            raise RuntimeError(f"Download finished but temp file {temp_path} not found.")

    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        logger.error(f"Failed to download {filename}: {e}")
        raise


if __name__ == "__main__":
    data_dir = Path("data")
    files_to_download = sys.argv[1:] if len(sys.argv) > 1 else ["d-d.csv", "abc-d.csv"]
    for f in files_to_download:
        download_zenodo_file(f, data_dir)
