# Mythologizer PostgreSQL Tests

This directory contains comprehensive tests for the `mythologizer_postgres` database module.

## Test Structure

### Unit Tests (`test_db.py`)
- **TestBuildUrl**: Tests environment variable reading and URL building
- **TestDatabaseConnections**: Tests connection functions (get_engine, get_session, session_scope, psycopg_connection)
- **TestSchemaFunctions**: Tests schema-related functions (_extract_schema_names, check_if_tables_exist)
- **TestDatabaseOperations**: Tests database operations (ping_db, get_table_row_counts, clear_all_rows)

### Integration Tests (`test_db_integration.py`)
- **TestDatabaseIntegration**: Tests with real database connection
  - Database connectivity
  - Schema application and table existence
  - Table structure validation
  - Data insertion, counting, and clearing
  - Vector operations
  - Complex myths table operations
  - Error handling
  - Concurrent operations

### Mytheme Store Tests (`test_mytheme_store.py`)
- **TestMythemeStore**: Tests for the mytheme store connector
  - Bulk insertion and retrieval of mythemes
  - Single insertion and retrieval operations
  - Mixed bulk and single operations
  - Retrieval by specific IDs
  - Error handling for non-existent IDs
  - Numpy vs list embedding formats
  - Large bulk operations (4 < n < 20 mythemes)
  - Embedding dimension consistency

### Myth Store Tests (`test_myth_store.py`)
- **TestMythStore**: Tests for the myth store connector with complex nested structures
  - Single myth insertion and retrieval with data integrity
  - Bulk myth insertion and retrieval with data integrity
  - Retrieval by specific myth IDs
  - Myth updating (single and bulk) with data integrity verification
  - Myth deletion (single and bulk)
  - Critical nested embedding precision testing
  - Large nested structures (4 < n < 20 myths with varying nested embeddings)
  - Complex data structures: main_embedding, embedding_ids, offsets (list of embeddings), weights

### Mythic Algebra Connector Tests (`test_mythic_algebra_connector.py`)
- **TestMythicAlgebraConnector**: Tests for the mythic algebra connector with myth matrices
  - Core mythicalgebra package function testing
  - Myth embedding retrieval (single and multiple)
  - Myth matrix composition and decomposition
  - Myth matrix retrieval with mytheme integration
  - Myth recalculation and updating with matrices
  - Complex myth matrix operations with larger structures
  - Integration with myth_store and mytheme_store
  - Matrix operations using the mythicalgebra package

## Running Tests

### Using Makefile (Recommended)
```bash
# Run all tests with fresh database
make test

# Run only unit tests
make test_setup

# Run only integration tests (requires database)
pytest tests/ -m integration
```

### Using pytest directly
```bash
# Run all tests
pytest tests/

# Run only unit tests
pytest tests/ -m unit

# Run only integration tests
pytest tests/ -m integration

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=mythologizer_postgres --cov-report=term-missing
```

### Using the test runner script
```bash
# Run all tests
python tests/run_tests.py

# Run only unit tests
python tests/run_tests.py --type unit

# Run only integration tests
python tests/run_tests.py --type integration

# Run with verbose output
python tests/run_tests.py --type all --verbose
```

## Test Requirements

### For Unit Tests
- pytest
- pytest-mock
- unittest.mock

### For Integration Tests
- All unit test requirements
- PostgreSQL database with pgvector extension
- Database connection (configured via `.env.test`)
- numpy (for vector operations)

### For Mytheme Store Tests
- All integration test requirements
- Real database connection for testing mytheme operations
- Bulk and single operation testing capabilities

### For Myth Store Tests
- All integration test requirements
- Real database connection for testing complex nested myth operations
- Support for complex data structures with multiple vector components
- Nested embedding and weight handling capabilities

### For Mythic Algebra Connector Tests
- All integration test requirements
- Real database connection for testing myth matrix operations
- Integration with myth_store and mytheme_store
- Support for mythicalgebra package operations
- Matrix composition, decomposition, and computation capabilities

## Test Configuration

Tests use the following configuration:
- Environment file: `.env.test`
- Database: PostgreSQL with pgvector extension
- Test database: `mythologizerdb_test`
- Test user: `test_user`
- Test password: `test`
- Embedding dimension: Read from `EMBEDDING_DIM` environment variable (default: 4 for tests)

## Test Coverage

The tests cover:

1. **Environment Variable Handling**
   - Reading database configuration from environment
   - Error handling for missing variables

2. **Database Connections**
   - SQLAlchemy engine creation
   - Session management
   - Context managers for transactions
   - Direct psycopg connections

3. **Schema Management**
   - Schema name extraction from SQL
   - Table existence checking
   - Schema application (automatic via Makefile)

4. **Database Operations**
   - Database connectivity testing
   - Row counting
   - Data clearing
   - Vector operations

5. **Integration Testing**
   - Real database operations
   - Vector similarity queries
   - Concurrent operations
   - Error handling

6. **Mytheme Store Operations**
   - Bulk insertion and retrieval of mythemes
   - Single mytheme operations
   - Mixed bulk and single operations
   - Retrieval by specific IDs
   - Error handling for non-existent records
   - Numpy vs list embedding format handling
   - Large-scale operations (4 < n < 20 mythemes)
   - Embedding dimension consistency across operations

7. **Myth Store Operations**
   - Complex nested data structure handling
   - Main embedding, embedding IDs, offsets (list of embeddings), and weights
   - Single and bulk myth operations
   - Myth updating and deletion operations
   - Critical nested embedding precision testing
   - Large-scale nested structures (4 < n < 20 myths with varying nested embeddings)
   - Data integrity verification for all vector components

8. **Mythic Algebra Connector Operations**
   - Myth matrix composition and decomposition using mythicalgebra package
   - Myth embedding computation and retrieval
   - Integration between myth_store and mytheme_store
   - Myth recalculation and updating with new matrices
   - Complex matrix operations with multiple mythemes
   - Matrix-based myth operations and transformations

## Troubleshooting

### Database Connection Issues
- Ensure PostgreSQL is running
- Check `.env.test` configuration
- Verify pgvector extension is installed
- Use `make fresh` to reset the test database

### Test Failures
- Unit tests should always pass (no external dependencies)
- Integration tests may fail if database is not available
- Check test output for specific error messages

### Environment Issues
- Ensure all required packages are installed: `uv sync --group dev`
- Check Python version compatibility
- Verify pytest is installed: `uv add --group dev pytest` 