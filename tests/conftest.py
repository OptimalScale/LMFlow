def pytest_addoption(parser):
    parser.addoption("--cpu-only", action="store_true", help="run tests that only requires cpu")
    parser.addoption("--skip-slow", action="store_true", help="skip slow tests")
    
def pytest_collection_modifyitems(config, items):
    new_items = []
    for item in items:
        func = item.function
        if config.getoption("--cpu-only"):
            if not (func.__doc__ and "#cpu" in func.__doc__.lower()):
                continue
        if config.getoption("--skip-slow"):
            if func.__doc__ and "#slow" in func.__doc__.lower():
                continue
        new_items.append(item)
    items[:] = new_items