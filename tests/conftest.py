from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def datadir():
    return Path(__file__).parent / "data"
