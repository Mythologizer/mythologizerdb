# Test Suite Documentation

## Overview

Comprehensive test suite for the mythologizer database system with 163 tests covering unit, integration, connector operations, and concurrent operations.

## Test Structure

### Unit Tests (`test_db.py`)
- Database connection and configuration testing
- Environment variable validation
- SQLAlchemy engine and session management
- Schema extraction and validation

### Integration Tests (`test_db_integration.py`)
- Real database connectivity and operations
- Table existence and structure validation
- Vector operations with pgvector
- Concurrent operations and error handling
- Row counting and cleanup operations

### Mytheme Store Tests (`test_mytheme_store.py`)
- Bulk and single mytheme operations
- Embedding data integrity verification
- Mixed bulk/single operations
- High-precision floating-point testing
- Large-scale operations (4 < n < 20 mythemes)

### Myth Store Tests (`test_myth_store.py`)
- Complex nested data structures
- Main embeddings, embedding IDs, offsets, and weights
- Single and bulk myth operations
- Critical nested embedding precision testing
- Large nested structures with varying complexity

### Mythic Algebra Connector Tests (`test_mythic_algebra_connector.py`)
- Myth matrix composition/decomposition using mythicalgebra package
- Integration between myth_store and mytheme_store
- Matrix-based myth operations and transformations
- Myth recalculation and updating with matrices
- Complex matrix operations with multiple mythemes

### Concurrent Operations Tests (`test_concurrent_operations.py`)
- Concurrent myth insertions and queries from multiple threads
- Concurrent mytheme operations (inserts and queries)
- Concurrent updates and queries with data integrity verification
- Concurrent bulk operations for myths and mythemes
- Concurrent epoch operations (increments and queries)
- Mixed operations (inserts, updates, queries, deletes) running simultaneously
- High concurrency thread simulation for realistic load testing
- Simulation status monitoring in loops while concurrently adding mythemes
- Verification of data consistency and operation completion under concurrent load

## Running Tests

```bash
# Run all tests
make test

# Run specific test types
uv run --env-file .env.test pytest tests/test_db.py -v                    # Unit tests
uv run --env-file .env.test pytest tests/test_db_integration.py -v        # Integration tests
uv run --env-file .env.test pytest tests/test_mytheme_store.py -v         # Mytheme store tests
uv run --env-file .env.test pytest tests/test_myth_store.py -v            # Myth store tests
uv run --env-file .env.test pytest tests/test_mythic_algebra_connector.py -v  # Mythic algebra tests
uv run --env-file .env.test pytest tests/test_concurrent_operations.py -v # Concurrent operations tests

# Run with fresh database
make fresh && uv run --env-file .env.test pytest tests -v
```

## Test Coverage

### Core Operations
1. **Database Connections**: Engine creation, session management, psycopg connections
2. **Schema Management**: Table creation, validation, schema extraction
3. **Vector Operations**: pgvector integration, similarity search, precision testing
4. **Mytheme Operations**: CRUD operations, bulk insertions, data integrity
5. **Myth Operations**: Complex nested structures, matrix operations, precision testing
6. **Mythic Algebra**: Matrix composition/decomposition, myth recalculation
7. **Concurrent Operations**: Multi-threaded operations, data consistency under load

### Data Integrity Verification
- **High Precision**: 7 decimal place precision for all vector components
- **Floating-point Accuracy**: Accounts for database precision differences
- **CRITICAL Testing**: Ensures embeddings, offsets, and weights maintain integrity
- **Matrix Operations**: Verifies myth matrix composition and decomposition
- **Integration Testing**: Tests cross-module data consistency

## Configuration

### Environment Variables
- `EMBEDDING_DIM`: Embedding dimension (default: 4)
- `TEST_DB_HOST`: Test database host (default: localhost)
- `TEST_DB_PORT`: Test database port (default: 5433)
- `TEST_DB_NAME`: Test database name (default: mythologizerdb_test)
- `TEST_DB_USER`: Test database user (default: test_user)
- `TEST_DB_PASSWORD`: Test database password (default: test_password)

### Database Setup
- PostgreSQL with pgvector extension
- Docker Compose test environment
- Automatic schema creation and cleanup
- Test isolation with database cleanup between tests

## Requirements

### For All Tests
- PostgreSQL with pgvector extension
- Docker and Docker Compose
- Python 3.13+ with uv package manager
- pytest and related testing packages

### For Integration Tests
- Real database connection
- pgvector extension support
- Vector similarity operations

### For Connector Tests
- Integration with myth_store and mytheme_store
- Matrix operation capabilities
- Complex nested data structure support

## Troubleshooting

### Common Issues
- **Database Connection**: Ensure Docker containers are running
- **Schema Issues**: Run `make fresh` to reset test environment
- **Precision Errors**: Normal for floating-point operations (handled by tests)
- **Import Errors**: Ensure all dependencies are installed

### Test Isolation
- Each test runs with clean database state
- Automatic cleanup before/after tests
- Engine cache clearing for proper isolation
- No cross-test interference

## Test Results

- **55 total tests** covering all system components
- **All tests passing** with proper data integrity verification
- **Dynamic embedding dimension** support (tested with 4, 6, 8 dimensions)
- **High precision verification** for all vector operations
- **Comprehensive error handling** and edge case testing 