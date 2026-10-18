import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "network: needs API keys and network; skipped by default")


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="network tests are opt-in: pytest -m network")
    if config.getoption("-m", default="") != "network":
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip)
