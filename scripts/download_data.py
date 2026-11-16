"""Download the small UCI dataset and extract only its three expected files."""

import hashlib
import io
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"


def main():
    target = ROOT / "data" / "raw"
    target.mkdir(parents=True, exist_ok=True)
    if any((target / name).exists() for name in ["bike-sharing.zip", "hour.csv", "day.csv", "Readme.txt"]):
        raise SystemExit("Data already exists; refusing to overwrite original files.")
    with urlopen(URL, timeout=60) as response:
        content = response.read(5_000_001)
    if len(content) > 5_000_000:
        raise SystemExit("Unexpected archive size; no files written.")
    with ZipFile(io.BytesIO(content)) as archive:
        files = {name: archive.read(name) for name in ["hour.csv", "day.csv", "Readme.txt"]}
    (target / "bike-sharing.zip").write_bytes(content)
    for name, data in files.items():
        (target / name).write_bytes(data)
    print("Archive SHA256:", hashlib.sha256(content).hexdigest())


if __name__ == "__main__":
    main()
