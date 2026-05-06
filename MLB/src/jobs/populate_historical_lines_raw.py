from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from common.contracts import require_columns
from jobs.build_training_artifacts import RAW_HISTORICAL_LINES_DIR
from odds.historical_lines import RAW_HISTORICAL_LINES_REQUIRED_COLUMNS


def _source_csv_paths(source: Path) -> list[Path]:
    if not source.exists():
        raise FileNotFoundError(f"Historical lines source does not exist: {source}")

    if source.is_file():
        if source.suffix.lower() != ".csv":
            raise ValueError(f"Historical lines source file must be a CSV: {source}")
        return [source]

    csv_paths = sorted(path for path in source.rglob("*.csv") if path.is_file())
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under historical lines source: {source}")
    return csv_paths


def _validate_raw_historical_lines_csv(path: Path) -> None:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    header_df = pd.DataFrame(columns=columns)
    require_columns(
        header_df,
        RAW_HISTORICAL_LINES_REQUIRED_COLUMNS,
        f"historical_lines_source[{path}]",
    )


def populate_historical_lines_raw(
    source: Path,
    *,
    target: Path = RAW_HISTORICAL_LINES_DIR,
) -> list[Path]:
    """
    Copy externally hydrated raw historical line CSVs into the local raw data tree.
    """
    source = source.resolve()
    target = target.resolve()

    copied_paths: list[Path] = []
    for csv_path in _source_csv_paths(source):
        _validate_raw_historical_lines_csv(csv_path)
        relative_path = csv_path.name if source.is_file() else csv_path.relative_to(source)
        target_path = target / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csv_path, target_path)
        copied_paths.append(target_path)

    return copied_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Populate MLB/data/raw/historical_lines/ from a local file or directory "
            "of externally hydrated CSV snapshots."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to a CSV file or directory containing raw historical lines CSVs.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=RAW_HISTORICAL_LINES_DIR,
        help="Destination raw historical lines directory. Defaults to the repo local raw path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    copied = populate_historical_lines_raw(args.source, target=args.target)

    print("Populated raw historical lines directory:")
    print(f"- source: {args.source.resolve()}")
    print(f"- target: {args.target.resolve()}")
    print(f"- copied_csv_files: {len(copied)}")
