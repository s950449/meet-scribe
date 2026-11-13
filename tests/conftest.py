from __future__ import annotations

import os
from pathlib import Path

import pytest

_LEAKY_PREFIXES = ("MEETSCRIBE_", "SLACK_", "OPENAI_")


@pytest.fixture(autouse=True)
def isolated_user_state(tmp_path_factory: pytest.TempPathFactory, monkeypatch) -> None:
    """Keep the developer's config, tokens, and outbox out of every test."""

    root: Path = tmp_path_factory.mktemp("user-state")
    for name in list(os.environ):
        if name.startswith(_LEAKY_PREFIXES):
            monkeypatch.delenv(name)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(root / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(root / "state"))
