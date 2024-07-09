import pytest

# With the following global module marker,
# monitoring is disabled by default:
pytestmark = [pytest.mark.monitor_skip_test]


def pytest_configure(config):
    """Set up pytest markers."""
    config.addinivalue_line("markers", "slowtest: mark test as slow")
