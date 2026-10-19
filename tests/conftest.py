import os

import pytest

# The model catalog (envforge/models/catalog.py) must never fetch during tests.
os.environ.setdefault("ENVFORGE_OFFLINE", "1")


def pytest_configure(config):
    config.addinivalue_line("markers", "network: needs API keys and network; skipped by default")
    config.addinivalue_line(
        "markers", "docker: needs a reachable Docker daemon; skipped automatically without one"
    )


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="network tests are opt-in: pytest -m network")
    if config.getoption("-m", default="") != "network":
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip)
