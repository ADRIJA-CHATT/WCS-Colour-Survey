from __future__ import annotations

from pathlib import Path
import os
import requests

from config import RAW_DIR, MAPPING_NAME, WCS_USE_FIXTURE, FIXTURE_DIR

# The WCS landing page is documentation. The maintained jvosten/wcs package
# points to the historical Berkeley files and also mirrors the core raw files
# on GitHub. GitHub is first because many modern environments cannot reach the
# old www1.icsi.berkeley.edu host.
SOURCES = {
    "term.txt": [
        "https://raw.githubusercontent.com/jvosten/wcs/master/data-raw/term.txt",
        "https://www1.icsi.berkeley.edu/wcs/data/20021219/txt/term.txt",
    ],
    "spkr.txt": [
        "https://raw.githubusercontent.com/jvosten/wcs/master/data-raw/spkr-lsas.txt",
        "https://www1.icsi.berkeley.edu/wcs/data/20100912/spkr-lsas.txt",
    ],
    # Language names are metadata only for this project. The analysis does not
    # require lang.txt, and the official HTML table is often inaccessible from
    # restricted compute environments. We therefore do NOT download it.
    MAPPING_NAME: [
        "https://raw.githubusercontent.com/jvosten/wcs/master/data-raw/cnum-vhcm-lab-new.txt",
        "https://www1.icsi.berkeley.edu/wcs/data/cnum-maps/cnum-vhcm-lab-new.txt",
    ],
}

DATA_URLS = {name: urls[0] for name, urls in SOURCES.items()}

ENV_NAMES = {
    "term.txt": "WCS_TERM_URL",
    "spkr.txt": "WCS_SPKR_URL",
    MAPPING_NAME: "WCS_MAPPING_URL",
}

HEADERS = {"User-Agent": "WCS-Color-Regression-Project/7.0"}


def _request(url: str) -> requests.Response:
    return requests.get(url, timeout=180, headers=HEADERS)


def _looks_like_html(content: bytes, content_type: str) -> bool:
    preview = content[:2000].lstrip().lower()
    return (
        "text/html" in content_type.lower()
        or preview.startswith(b"<!doctype")
        or preview.startswith(b"<html")
        or preview.startswith(b"<?xml")
    )


def _valid_data_payload(filename: str, content: bytes, content_type: str) -> bool:
    if not content:
        return False
    if _looks_like_html(content, content_type):
        return False

    text = content[:5000].decode("utf-8", errors="replace")
    lines = [x for x in text.splitlines() if x.strip() and not x.lstrip().startswith("#")]

    if filename == "term.txt":
        parts = lines[0].split("\t") if lines else []
        return len(parts) == 4 and parts[0].strip().isdigit() and parts[2].strip().isdigit()
    if filename == "spkr.txt":
        parts = lines[0].split("\t") if lines else []
        return len(parts) >= 4 and parts[0].strip().isdigit() and parts[1].strip().isdigit()
    if filename == MAPPING_NAME:
        # The raw mapping file has either a header containing L*/a*/b* or
        # numeric rows with at least 9 whitespace-delimited fields.
        if "L*" in text and "a*" in text and "b*" in text:
            return True
        return any(len(line.split()) >= 9 and line.split()[0].isdigit() for line in lines)
    return True


def _download_from_candidates(filename: str, destination: Path) -> str:
    override = os.environ.get(ENV_NAMES[filename])
    candidates = [override] if override else SOURCES[filename]
    errors: list[str] = []

    for url in candidates:
        print(f"Downloading WCS file: {url}")
        try:
            response = _request(url)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if not _valid_data_payload(filename, response.content, content_type):
                errors.append(f"{url}: invalid payload for {filename}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(response.content)
            print(f"  saved {destination} ({destination.stat().st_size:,} bytes)")
            return url
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    raise RuntimeError(
        f"Could not obtain real WCS file {filename}. Tried:\n"
        + "\n".join(f"  - {e}" for e in errors)
        + "\n\nYou can override the source with the corresponding WCS_*_URL environment variable."
    )


def _copy_fixture() -> None:
    raw_dir = RAW_DIR / "wcs_archive"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["term.txt", "spkr.txt", "lang.txt", MAPPING_NAME]:
        src = FIXTURE_DIR / filename
        dst = raw_dir / filename
        dst.write_bytes(src.read_bytes())


def _valid_local(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    preview = path.read_bytes()[:5000]
    return not _looks_like_html(preview, "")


def ensure_wcs_data(force: bool = False) -> None:
    if WCS_USE_FIXTURE:
        _copy_fixture()
        return

    raw_dir = RAW_DIR / "wcs_archive"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Only these three files are required for the two statistical analyses.
    # lang.txt is optional metadata and is deliberately not downloaded.
    for filename in ["term.txt", "spkr.txt", MAPPING_NAME]:
        path = raw_dir / filename
        if force or not _valid_local(path):
            if path.exists():
                path.unlink()
            _download_from_candidates(filename, path)


def main() -> None:
    ensure_wcs_data(force=os.environ.get("WCS_FORCE_DOWNLOAD", "0") == "1")
    print("WCS core data are ready.")


if __name__ == "__main__":
    main()
