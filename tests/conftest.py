import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires gpu")
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "lmflow_core: tests for core lmflow functionality")