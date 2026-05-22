from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = REPO_ROOT / "vendor" / "pop-apple2" / "01 POP Source" / "Source"


@pytest.fixture(scope="session")
def source_dir() -> Path:
    if not SOURCE_DIR.is_dir():
        pytest.skip(f"submodule not checked out: {SOURCE_DIR}")
    return SOURCE_DIR
