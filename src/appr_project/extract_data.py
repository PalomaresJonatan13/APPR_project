"""Convert the training PSV archives into one CSV per training set."""

from __future__ import annotations

import argparse
import csv
import io
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Sequence


_PATIENT_FILENAME = re.compile(r"p(\d+)\.psv$", re.IGNORECASE)
_SETS = ("A", "B")


def convert_archive(archive_path: Path, output_path: Path, set_name: str) -> tuple[int, int]:
    """Write all records in one PSV archive to CSV and return patient and row counts."""
    if set_name not in _SETS:
        raise ValueError(f"Unsupported training set: {set_name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    patient_count = 0
    row_count = 0
    header: list[str] | None = None
    seen_patient_ids: set[str] = set()

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=output_path.parent,
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            with zipfile.ZipFile(archive_path) as archive:
                members = [
                    (member, _patient_number(member.filename))
                    for member in archive.infolist()
                    if not member.is_dir() and member.filename.lower().endswith(".psv")
                ]
                members.sort(key=lambda item: item[1])
                writer = csv.writer(temporary_file)

                for member, patient_number in members:
                    patient_id = f"{set_name}_{patient_number:06d}"
                    if patient_id in seen_patient_ids:
                        raise ValueError(f"Duplicate patient ID {patient_id} in {archive_path}")
                    seen_patient_ids.add(patient_id)

                    with archive.open(member) as binary_file:
                        with io.TextIOWrapper(binary_file, encoding="utf-8", newline="") as psv_file:
                            reader = csv.reader(psv_file, delimiter="|")
                            file_header = next(reader, None)
                            if not file_header:
                                raise ValueError(f"Missing header in {member.filename}")
                            if header is None:
                                header = [*file_header, "patient_id", "source_set"]
                                writer.writerow(header)
                            elif file_header != header[:-2]:
                                raise ValueError(f"Inconsistent PSV header in {member.filename}")

                            for line_number, row in enumerate(reader, start=2):
                                if len(row) != len(file_header):
                                    raise ValueError(
                                        f"Expected {len(file_header)} fields in "
                                        f"{member.filename}:{line_number}, got {len(row)}"
                                    )
                                writer.writerow([*row, patient_id, set_name])
                                row_count += 1
                    patient_count += 1

            if header is None:
                raise ValueError(f"No PSV files found in {archive_path}")
            temporary_file.flush()
            temporary_file.close()
            temporary_path.replace(output_path)
        except Exception:
            temporary_file.close()
            temporary_path.unlink(missing_ok=True)
            raise

    return patient_count, row_count


def _patient_number(member_name: str) -> int:
    match = _PATIENT_FILENAME.fullmatch(PurePosixPath(member_name).name)
    if match is None:
        raise ValueError(f"Unexpected PSV filename: {member_name}")
    return int(match.group(1))


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Combine the training PSV archives into set_A.csv and set_B.csv."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data",
        help="Data directory containing raw/ and csv/ (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    raw_dir = args.data_dir / "raw"
    output_dir = args.data_dir / "csv"
    for set_name in _SETS:
        archive_path = raw_dir / f"training_set{set_name}.zip"
        output_path = output_dir / f"set_{set_name}.csv"
        patients, rows = convert_archive(archive_path, output_path, set_name)
        print(f"Set {set_name}: {patients:,} patients, {rows:,} rows -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
