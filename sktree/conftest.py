def pytest_configure(config):
    """Set up pytest markers."""
    config.addinivalue_line("markers", "slowtest: mark test as slow")
