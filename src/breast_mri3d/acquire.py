"""Pinned acquisition and patient-level allocation for the expert I-SPY1 release."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from tempfile import NamedTemporaryFile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


ISPY_ACCESSION = "10.7937/TCIA.XC7A-QT20"
ISPY_MIRROR = "https://huggingface.co/datasets/MedOtter/ISPY1-Tumor-SEG-Radiomics/resolve"
ISPY_REVISION = "dd4bbffd69294daab3bfdc1999625313578f4c2c"
ISPY_SUMS = "ISPY1-Tumor-SEG-Radiomics.sums"

AMBL_URL = "https://zenodo.org/records/17253223/files"
AMBL_DCE = "BreastDCEDL_AMBL_dce.tar.gz"
AMBL_DCE_SIZE = 15_456_276_546
AMBL_ASSETS = {
    "BreastDCEDL_AMBL_paitents.csv": "5cbed21663d18eed63fdd83dfb199ac0",
    "sus_tum.tar.gz": "14d77cdc88cb702ff5f46fdc94a9a291",
    AMBL_DCE: "778b0f8eb33e69d0a2c3d39b470be1e4",
}


def checksum(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_relative(value: str) -> PurePosixPath:
    if "\\" in value or ":" in value:
        raise ValueError("Expected an archive-style relative path.")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("Unsafe archive path.")
    return path


def download(url: str, destination: Path, *, expected: str | None = None, algorithm: str = "md5") -> Path:
    """Write an incomplete transfer separately and verify known publisher hashes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected is not None and checksum(destination, algorithm) != expected:
            raise ValueError(f"Existing {destination.name} differs from the publisher checksum.")
        return destination
    partial = destination.with_name(destination.name + ".partial")
    if expected is not None and partial.exists() and checksum(partial, algorithm) == expected:
        partial.replace(destination)
        return destination
    for attempt in range(4):
        count = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "Breast-MRI-3D-research/0.2", "Accept-Encoding": "identity"}
        if count:
            headers["Range"] = f"bytes={count}-"
        try:
            with urlopen(Request(url, headers=headers), timeout=45) as response:
                resumed = response.status == 206
                if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {count}-"):
                    raise OSError("Remote server returned an unexpected byte range.")
                with partial.open("ab" if resumed else "wb") as stream:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(block)
                length = response.headers.get("Content-Length")
                if length and partial.stat().st_size != (count if resumed else 0) + int(length):
                    raise OSError("Transfer ended before its stated length.")
            if expected is not None and checksum(partial, algorithm) != expected:
                raise ValueError(f"Publisher checksum mismatch for {destination.name}.")
            partial.replace(destination)
            return destination
        except HTTPError as exc:
            if exc.code not in {408, 429, 500, 502, 503, 504} or attempt == 3:
                raise
        except (URLError, OSError):
            if attempt == 3:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable download retry state.")


def read_md5_manifest(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([a-fA-F0-9]{32})\s+\*?(.+)", line.strip())
        if match is None:
            raise ValueError("Invalid publisher MD5 manifest line.")
        relative = match[2].removeprefix("./").removeprefix("ISPY1-Tumor-SEG-Radiomics/")
        canonical = safe_relative(relative).as_posix()
        if canonical in entries and entries[canonical] != match[1].lower():
            raise ValueError("Conflicting file checksums in the publisher manifest.")
        entries[canonical] = match[1].lower()
    return entries


def ispy_id(value: str) -> str:
    matched = re.fullmatch(r"ISPY_?1_(\d+)", value.strip(), re.IGNORECASE)
    if not matched:
        raise ValueError("Unrecognized I-SPY1 identity.")
    return f"ISPY1_{int(matched[1])}"


def allocate_ispy(rows: list[dict], sums: dict[str, str], known: set[str], limit: int, seed: int) -> list[dict]:
    if limit < 5:
        raise ValueError("Reserve at least five cases for the patient-level study.")
    eligible = {}
    for row in rows:
        patient = ispy_id(row["patient_id"])
        if patient in eligible:
            raise ValueError("Duplicate I-SPY1 subject in the upstream manifest.")
        phases = row["images_native"][:2]
        mask = row["mask"]
        if len(phases) != 2 or not mask.startswith("NIfTI-Files/masks_stv_manual/"):
            continue
        paths = [safe_relative(name).as_posix() for name in [*phases, mask]]
        if not all(name in sums for name in paths):
            continue
        if (not re.search(r"_0000(?:[_.]|$)", PurePosixPath(paths[0]).name)
                or not re.search(r"_0001(?:[_.]|$)", PurePosixPath(paths[1]).name)
                or any(patient not in PurePosixPath(name).name for name in paths)):
            raise ValueError("The public manifest does not match its documented phase or identity mapping.")
        eligible[patient] = {"patient": patient, "files": dict(zip(("pre", "post", "lesion"), paths)),
                             "md5": {name: sums[name] for name in paths}}
    known = {ispy_id(patient) for patient in known}
    order = sorted(eligible, key=lambda p: (p not in known, hashlib.sha256(f"{seed}:{p}".encode()).digest()))
    selected = order[:limit]
    validation_count = max(1, round(.2 * len(selected)))
    unseen = [patient for patient in selected if patient not in known]
    if len(unseen) < validation_count:
        raise ValueError("Too few previously unseen patients to lock the validation partition.")
    validation = set(unseen[:validation_count])
    return [{**eligible[patient], "split": "validation" if patient in validation else "train"}
            for patient in selected]


def acquire_ispy(root: Path, legacy: Path, *, limit: int = 120, seed: int = 19, workers: int = 8) -> Path:
    meta = root / "acquisition"
    raw = root / "raw" / "ISPY1-STV"
    base = f"{ISPY_MIRROR}/{ISPY_REVISION}"
    checksums = download(f"{base}/{ISPY_SUMS}", meta / ISPY_SUMS)
    upstream = download(f"{base}/train.jsonl", meta / "ispy1-upstream.jsonl")
    rows = [json.loads(line) for line in upstream.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    known = {row["patient"] for row in json.loads(legacy.read_text(encoding="utf-8-sig"))["cases"]}
    cases = allocate_ispy(rows, read_md5_manifest(checksums), known, limit, seed)
    locked = meta / "ispy1-stv-selection.json"
    planned = {"dataset": ISPY_ACCESSION, "revision": ISPY_REVISION, "seed": seed,
               "publisher_manifest_sha256": checksum(checksums), "cases": cases}
    if locked.exists() and json.loads(locked.read_text(encoding="utf-8")) != planned:
        raise ValueError("The requested selection differs from the previously locked experiment.")
    write_json(locked, planned)

    def acquire_case(case):
        for name in case["files"].values():
            destination = raw.joinpath(*safe_relative(name).parts)
            download(f"{base}/{quote(name, safe='/')}", destination, expected=case["md5"][name])
        return case

    ready, failed = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(acquire_case, case): case for case in cases}
        for future in as_completed(futures):
            try:
                ready.append(future.result())
            except (ValueError, OSError) as exc:
                failed.append({"patient": futures[future]["patient"], "error": str(exc)})
            if (len(ready) + len(failed)) % 10 == 0:
                print(f"I-SPY1 STV verified {len(ready)}/{len(cases)}, transfer failures {len(failed)}", flush=True)
    ledger = meta / "ispy1-stv-acquired.json"
    write_json(ledger, {**planned, "cases": sorted(ready, key=lambda c: c["patient"]), "failed": failed})
    if not ready:
        raise RuntimeError("No downloaded expert I-SPY1 patient passed integrity checks.")
    return ledger


def download_ranges(url: str, target: Path, length: int, expected: str, *, workers: int = 4) -> None:
    """Assemble a resumable large transfer using validated HTTP byte ranges."""
    if target.exists():
        if checksum(target, "md5") != expected:
            raise ValueError("Existing archive failed its version checksum.")
        return
    block_size = 256 * 1024 ** 2
    ranges = [(offset, min(length, offset + block_size)) for offset in range(0, length, block_size)]
    directory = target.parent / ".dce-ranges"
    directory.mkdir(parents=True, exist_ok=True)
    conventional = target.with_name(target.name + ".partial")
    prefix = conventional.stat().st_size if conventional.is_file() else 0

    def block(bounds: tuple[int, int]) -> Path:
        start, end = bounds
        part = directory / f"{start:012d}.block"
        if part.is_file() and part.stat().st_size == end - start:
            return part
        partial = part.with_suffix(".partial")
        if end <= prefix:
            try:
                with conventional.open("rb") as source, partial.open("wb") as stream:
                    source.seek(start)
                    remain = end - start
                    while remain:
                        contents = source.read(min(remain, 1024 * 1024))
                        if not contents:
                            raise OSError("The earlier transfer stopped inside this block.")
                        stream.write(contents)
                        remain -= len(contents)
                partial.replace(part)
                return part
            except OSError:
                pass
        headers = {"User-Agent": "Breast-MRI-3D-research/0.2", "Accept-Encoding": "identity",
                   "Range": f"bytes={start}-{end - 1}"}
        for attempt in range(4):
            try:
                with urlopen(Request(url, headers=headers), timeout=35) as response:
                    if (response.status != 206
                            or response.headers.get("Content-Range") != f"bytes {start}-{end - 1}/{length}"):
                        raise ValueError("The archive server did not honor the exact requested byte range.")
                    with partial.open("wb") as stream:
                        for contents in iter(lambda: response.read(1024 * 1024), b""):
                            stream.write(contents)
                if partial.stat().st_size != end - start:
                    raise OSError("The archive block was truncated.")
                partial.replace(part)
                return part
            except (URLError, OSError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("Unreachable range retry state.")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(block, bounds) for bounds in ranges]
        for count, future in enumerate(as_completed(jobs), start=1):
            future.result()
            if count % 5 == 0:
                print(f"AMBL DCE byte blocks received {count}/{len(ranges)}", flush=True)
    temporary = target.with_name(target.name + ".assembled")
    digest = hashlib.md5()
    with temporary.open("wb") as stream:
        for future in jobs:
            with future.result().open("rb") as source:
                for contents in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    digest.update(contents)
                    stream.write(contents)
    if temporary.stat().st_size != length or digest.hexdigest() != expected:
        raise ValueError("The assembled archive failed the publisher MD5 check.")
    temporary.replace(target)


def _adopt_ambl_archive(source: Path, destination: Path) -> None:
    """Verify the user's download before committing a copy to the dataset directory."""
    if source.stat().st_size != AMBL_DCE_SIZE:
        raise ValueError("The local AMBL archive size differs from Zenodo v3.")
    digest = AMBL_ASSETS[AMBL_DCE]
    if destination.exists():
        if checksum(destination, "md5") != digest:
            raise ValueError("The existing project archive fails the Zenodo v3 checksum.")
        return
    with NamedTemporaryFile(dir=destination.parent, prefix=".ambl-import-", delete=False) as stream:
        staged = Path(stream.name)
        try:
            fingerprint = hashlib.md5()
            with source.open("rb") as original:
                for block in iter(lambda: original.read(4 * 1024 * 1024), b""):
                    fingerprint.update(block)
                    stream.write(block)
            if (fingerprint.hexdigest() != digest or stream.tell() != AMBL_DCE_SIZE
                    or source.stat().st_size != AMBL_DCE_SIZE):
                raise ValueError("The local AMBL archive fails the Zenodo v3 checksum.")
            stream.close()
            staged.rename(destination)
        finally:
            stream.close()
            staged.unlink(missing_ok=True)


def acquire_ambl(root: Path, *, ranged: bool = False, local_dce: Path | None = None) -> Path:
    directory = root / "raw" / "AMBL"
    directory.mkdir(parents=True, exist_ok=True)
    if local_dce is not None and ranged:
        raise ValueError("Choose a local archive or a ranged download.")
    for name, digest in AMBL_ASSETS.items():
        print(f"AMBL verify or acquire {name}", flush=True)
        url = f"{AMBL_URL}/{name}?download=1"
        if local_dce is not None and name == AMBL_DCE:
            _adopt_ambl_archive(local_dce, directory / name)
        elif ranged and name == AMBL_DCE:
            download_ranges(url, directory / name, AMBL_DCE_SIZE, digest)
        else:
            download(url, directory / name, expected=digest)
    proof = directory / "archive-integrity.json"
    write_json(proof, {"record": "zenodo:17253223/v3", "purpose": "reserved_external",
                       "files": [{"name": name, "md5": h, "bytes": (directory / name).stat().st_size}
                                 for name, h in AMBL_ASSETS.items()]})
    return proof
