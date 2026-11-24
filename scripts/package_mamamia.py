"""Build a private MAMA-MIA Task 1 archive from a frozen evaluation lock."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from zipfile import ZIP_DEFLATED, ZipFile

from breast_mri3d.acquire import checksum, write_json


REPO = Path(__file__).resolve().parents[1]
MODULES = ("__init__", "acquire", "cascade", "dicom", "geometry", "imaging", "inference", "labels",
           "network", "prepare", "research")


def build(root: Path, lock_path: Path, output: Path) -> None:
    root = root.resolve()
    freeze = json.loads(lock_path.read_text(encoding="utf-8"))
    proposed = freeze["proposed"]
    limits = {"none": None, "raw": None, "largest": 1, "two_largest": 2}
    if proposed["postprocessing"] not in limits or not 0 < proposed["threshold"] < 1:
        raise ValueError("Unsupported frozen inference configuration.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.zip")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(REPO / "scripts/mamamia_task1.py", "model.py")
        archive.write(REPO / "LICENSE", "LICENSE")
        archive.writestr("requirements.txt", "numpy\nscipy\nnibabel\npydicom\n")
        for module in MODULES:
            name = f"breast_mri3d/{module}.py"
            archive.write(REPO / "src" / name, name)
        locator = proposed.get("locator")
        for name, record in (("compact.pt", proposed["model"]), ("locator.pt", locator)):
            if record is not None:
                path = (root / record["path"]).resolve()
                if not path.is_relative_to(root) or checksum(path) != record["sha256"]:
                    raise ValueError("A candidate checkpoint differs from the frozen evaluation.")
                archive.write(path, name)
        archive.writestr("operating-point.json", json.dumps(
            {"threshold": proposed["threshold"], "checkpoint_sha256": proposed["model"]["sha256"],
             "locator_sha256": locator["sha256"] if locator else None,
             "patch_zyx": proposed.get("patch_zyx"), "mirror_left_right": proposed.get("mirror_left_right", False),
             "component_limit": limits[proposed["postprocessing"]]}, indent=2))
    os.replace(temporary, output)
    write_json(output.with_suffix(".json"), {"sha256": checksum(output), "evaluation_lock_sha256": checksum(lock_path),
               "task": 1, "submission": "local"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--wait-hours", type=float, default=0.)
    args = parser.parse_args()
    lock = args.root / "runs/v03/evaluation-lock.json"
    deadline = time.monotonic() + args.wait_hours * 3600
    while not lock.is_file() and time.monotonic() < deadline:
        time.sleep(30)
    build(args.root, lock, args.root / "benchmark/MAMA-MIA-task1-v03.zip")


if __name__ == "__main__":
    main()
