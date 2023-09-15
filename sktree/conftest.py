def pytest_configure(config):
    config.addinivalue_line("markers", "slowtest: mark test as slow")
