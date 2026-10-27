"""Shared test settings: every test writes its run log to a temporary file, never to logs/run.jsonl."""

import pytest

from erasmus_agent import logbook


@pytest.fixture(autouse=True)
def temporary_run_log(monkeypatch, tmp_path):
    monkeypatch.setattr(logbook, "LOG_FILE", tmp_path / "run.jsonl")
    return tmp_path / "run.jsonl"
