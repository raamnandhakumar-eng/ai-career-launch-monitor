"""
Download the Anthropic Economic Index labor-market-impact files.

Pulls `job_exposure.csv` (occupation-level observed AI exposure) and
`task_penetration.csv` (O*NET task-level penetration) from the public
Hugging Face dataset (CC-BY).

    python -m src.download_anthropic            # both files
    python -m src.download_anthropic --exposure-only

Note: `job_exposure.csv` is already vendored in data/raw/anthropic/ so the
descriptive analysis runs out of the box. Re-running this refreshes it to the
latest release and adds the larger task_penetration.csv.
"""
import argparse
import urllib.request

from src.config import AEI_FILES, RAW_ANTHROPIC


def _download(url: str, dest):
    print(f"  -> {dest.name} ... ", end="", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "ai-career-launch-monitor"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        f.write(r.read())
    print(f"{dest.stat().st_size/1e3:,.0f} kB")


def main(exposure_only: bool = False):
    print("Downloading Anthropic Economic Index (labor market impacts):")
    for name, url in AEI_FILES.items():
        if exposure_only and name != "job_exposure.csv":
            continue
        _download(url, RAW_ANTHROPIC / name)
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposure-only", action="store_true",
                    help="skip the large task_penetration.csv (1.9 MB)")
    main(**vars(ap.parse_args()))
