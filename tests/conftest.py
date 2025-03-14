import pytest

def pytest_configure(config: pytest.Config):
    config.addinivalue_line("markers", "gpu: requires gpu")
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "lmflow_core: tests for core lmflow functionality")
    config.addinivalue_line("markers", "dothis: mark for dev to do the specified tests only")