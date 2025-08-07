# Mythologizer PostgreSQL

A PostgreSQL database connector for mythology data with vector embeddings, built with SQLAlchemy and pgvector.

## Installation

### Using uv (recommended)

```bash
# Install in development mode
uv pip install -e .

# Or install from PyPI (when published)
uv add mythologizer_postgres
```

### Using pip

```bash
pip install mythologizer_postgres
```

## Package Structure

The package is organized into the following structure:

```
mythologizer_postgres/
├── __init__.py          # Core database functions
├── db.py               # Database connection and utilities
├── schema.py           # Database schema definitions
├── cli.py              # Command-line interface
└── connectors/         # Data access layer
    ├── __init__.py     # All connector functions
    ├── myth_store.py   # Myth data operations
    ├── mytheme_store.py # Mytheme data operations
    └── mythicalgebra/  # Myth algebra subpackage
        ├── __init__.py # Myth algebra functions
        └── mythic_algebra_connector.py # Myth algebra operations
```

## Usage

### Core Database Functions

Direct imports from the main package:

```python
from mythologizer_postgres import (
    get_engine,
    get_session,
    session_scope,
    psycopg_connection,
    apply_schemas,
    check_if_tables_exist,
    ping_db,
    clear_all_rows,
    get_table_row_counts,
    MissingEnvironmentVariable,
    need,
    build_url,
)

# Example: Get a database session
with session_scope() as session:
    # Your database operations here
    pass

# Example: Check database connectivity
if ping_db():
    print("Database is accessible")
```

### Connector Functions

Import all connector functions:

```python
from mythologizer_postgres.connectors import (
    # Mytheme functions
    get_mythemes_bulk,
    get_mytheme,
    insert_mythemes_bulk,
    
    # Myth functions
    insert_myth,
    insert_myths_bulk,
    get_myth,
    get_myths_bulk,
    update_myth,
    update_myths_bulk,
    delete_myth,
    delete_myths_bulk,
    
    # Myth algebra functions
    get_myth_embeddings,
    get_myth_matrices,
    recalc_and_update_myths,
)

# Example: Get a myth by ID
myth = get_myth(123)

### Myth Algebra Functions

Import myth algebra functions specifically:

```python
from mythologizer_postgres.connectors.mythicalgebra import (
    get_myth_embeddings,
    get_myth_matrices,
    recalc_and_update_myths,
)

# Example: Get myth embeddings
embeddings = get_myth_embeddings([123, 124, 125])

# Example: Get myth matrices
matrices = get_myth_matrices([123, 124, 125])

# Example: Recalculate and update myths
updated_ids = recalc_and_update_myths([123, 124, 125])
```

### Subpackage Access

You can also access the subpackages directly:

```python
import mythologizer_postgres.connectors as connectors
import mythologizer_postgres.connectors.mythicalgebra as mythicalgebra

# Use the functions
myth = connectors.get_myth(123)
embedding = mythicalgebra.get_myth_embeddings(123)
```

## Environment Variables

The package requires the following environment variables:

```bash
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=your_database
```

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd mythologizerDB

# Install in development mode
uv pip install -e .

# Run tests
uv run python test_imports.py
```

### Building the Package

```bash
# Build source distribution and wheel
uv build