# Myth Insertion Performance Benchmark

This benchmark script measures the performance difference between single myth insertions and bulk myth insertions.

## What it measures

The script compares:
- **Single insertions**: Inserting myths one by one using `insert_myth()`
- **Bulk insertions**: Inserting multiple myths at once using `insert_myths_bulk()`

## Test scenarios

The benchmark tests the following myth counts:
- 10 myths
- 20 myths  
- 50 myths
- 100 myths
- 200 myths

Each test runs 3 times to get average performance metrics.

## How to run

### Option 1: Fresh database setup (recommended)
```bash
make benchmark
```

This will:
1. Start a fresh test database
2. Apply schemas
3. Run the benchmark
4. Clean up

### Option 2: Use existing database
```bash
make benchmark-quick
```

This assumes the test database is already running and will run the benchmark directly.

### Option 3: Run manually
```bash
# Ensure test database is running first
make fresh

# Then run the benchmark
uv run --env-file .env.test python mythologizer_postgres/benchmark.py
```

## Output

The benchmark produces a detailed table showing:
- **Count**: Number of myths inserted in each test
- **Single (s)**: Total time to insert all myths one by one (average of 3 runs)
- **Bulk (s)**: Total time to insert all myths in bulk (average of 3 runs)
- **Speedup**: How many times faster bulk insertion is compared to single insertion
- **Single/rec**: Average time per individual myth insertion
- **Bulk/rec**: Average time per myth when using bulk insertion

## Example output

```
================================================================================
BENCHMARK RESULTS
================================================================================
Count    Single (s)       Bulk (s)        Speedup      Single/rec        Bulk/rec        
--------------------------------------------------------------------------------
10       0.1234           0.0456           2.71x        0.012340          0.004560        
20       0.2345           0.0789           2.97x        0.011725          0.003945        
50       0.5678           0.1567           3.62x        0.011356          0.003134        
100      1.1234           0.2987           3.76x        0.011234          0.002987        
200      2.3456           0.5678           4.13x        0.011728          0.002839        
--------------------------------------------------------------------------------
```

## Configuration

The benchmark uses the following environment variables:
- `EMBEDDING_DIM`: Dimension of embeddings (default: 4)
- Database connection variables from `.env.test`

## Test data

Each myth contains:
- A main embedding vector
- 2-5 nested embeddings with random IDs
- Random offset vectors
- Normalized weights that sum to 1.0

The data is generated fresh for each test run to ensure consistent benchmarking.
