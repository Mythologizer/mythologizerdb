import pytest
import os
from unittest.mock import patch


@pytest.fixture(scope="session")
def test_env():
    """Fixture to ensure test environment variables are set."""
    # These should match your .env.test file
    test_env_vars = {
        'POSTGRES_USER': 'test_user',
        'POSTGRES_PASSWORD': 'test',
        'POSTGRES_DB': 'mythologizerdb_test',
        'POSTGRES_HOST': 'localhost',
        'POSTGRES_PORT': '5433',
        'EMBEDDING_DIM': os.getenv('EMBEDDING_DIM', '4')  # Read from environment or default to 4
    }
    
    # Set environment variables for testing
    with patch.dict(os.environ, test_env_vars):
        yield test_env_vars


@pytest.fixture(scope="function")
def clean_database():
    """Fixture to clean the database before each test."""
    from mythologizer_postgres.db import clear_all_rows, get_engine
    
    # Clear the engine cache to ensure fresh connections
    get_engine.cache_clear()
    
    # Clean up before test
    try:
        clear_all_rows()
    except Exception:
        pass  # Database might not be available
    
    yield
    
    # Clean up after test
    try:
        clear_all_rows()
    except Exception:
        pass  # Database might not be available
    
    # Clear the engine cache again
    get_engine.cache_clear()


@pytest.fixture(scope="session")
def database_available():
    """Fixture to check if database is available for integration tests."""
    from mythologizer_postgres.db import ping_db, get_engine
    
    # Clear the engine cache to ensure we get a real connection
    get_engine.cache_clear()
    
    try:
        is_available = ping_db()
        if not is_available:
            pytest.skip("Database is not available")
        return is_available
    except Exception:
        pytest.skip("Database connection failed")


@pytest.fixture(autouse=True)
def clear_engine_cache():
    """Automatically clear engine cache before each test to prevent mock interference."""
    from mythologizer_postgres.db import get_engine
    get_engine.cache_clear()


# Mark tests that require database
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring database"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test (no database required)"
    )


# Skip integration tests if database is not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip integration tests if database is not available."""
    from mythologizer_postgres.db import ping_db, get_engine
    
    # Clear engine cache before checking database availability
    get_engine.cache_clear()
    
    try:
        db_available = ping_db()
    except Exception:
        db_available = False
    
    skip_integration = pytest.mark.skip(reason="Database not available for integration test")
    
    for item in items:
        if "integration" in item.keywords and not db_available:
            item.add_marker(skip_integration) 