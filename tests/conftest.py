import pytest

from pathlib import Path


@pytest.fixture(scope="session")
def datadir():
    return Path(__file__).parent / "data"
