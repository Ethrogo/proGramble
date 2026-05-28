import pandas as pd
import pytest

from pitcher_k import data_loader


def test_load_statcast_data_splits_failed_chunk_and_recovers(monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_statcast(*, start_dt: str, end_dt: str, parallel: bool):
        calls.append((start_dt, end_dt))
        if (start_dt, end_dt) == ("2023-06-26", "2023-07-02"):
            raise pd.errors.ParserError("Error tokenizing data. C error: Expected 1 fields")

        return pd.DataFrame(
            [
                {
                    "game_date": start_dt,
                    "game_pk": len(calls),
                    "pitcher": 1,
                }
            ]
        )

    monkeypatch.setattr(data_loader, "statcast", fake_statcast)
    monkeypatch.setattr(data_loader.time, "sleep", lambda _: None)

    result = data_loader.load_statcast_data(
        "2023-06-26",
        "2023-07-02",
        chunk_days=7,
        max_retries=2,
    )

    assert calls[:2] == [
        ("2023-06-26", "2023-07-02"),
        ("2023-06-26", "2023-07-02"),
    ]
    assert ("2023-06-26", "2023-06-29") in calls
    assert ("2023-06-30", "2023-07-02") in calls
    assert len(result) == 2
    assert result["game_date"].tolist() == [
        "2023-06-26",
        "2023-06-30",
    ]


def test_load_statcast_data_raises_for_single_day_failure(monkeypatch):
    def fake_statcast(*, start_dt: str, end_dt: str, parallel: bool):
        raise pd.errors.ParserError(f"bad payload for {start_dt} to {end_dt}")

    monkeypatch.setattr(data_loader, "statcast", fake_statcast)
    monkeypatch.setattr(data_loader.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="Failed Statcast chunk after retries: 2023-06-26 to 2023-06-26"):
        data_loader.load_statcast_data(
            "2023-06-26",
            "2023-06-26",
            chunk_days=1,
            max_retries=2,
        )
