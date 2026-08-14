"""
Download O*NET inputs (Job Zones + Education/Training/Experience).

Used to characterise entry-level dependence and education pathways for the
policy layer (apprenticeships, curriculum, retraining) and as an alternative
"entry-level dependence" signal in the ELSI.

    python -m src.download_onet

O*NET codes are SOC-based ("15-1252.00"); truncate at 7 chars to join on SOC.
"""
import urllib.request

from src.config import ONET_JOBZONES, ONET_EDU, RAW_ONET


def _get(url: str, name: str):
    print(f"  -> {name} ... ", end="", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "ai-career-launch-monitor"})
    with urllib.request.urlopen(req) as r, open(RAW_ONET / name, "wb") as f:
        f.write(r.read())
    print("ok")


def main():
    print("Downloading O*NET files:")
    _get(ONET_JOBZONES, "job_zones.txt")
    _get(ONET_EDU, "education.txt")
    print(f"Done -> {RAW_ONET}")


if __name__ == "__main__":
    main()
