#!/usr/bin/env python3
"""
Performance benchmarking script for myth insertions.
Measures single vs bulk insertion performance for different myth counts.
"""

import time
import numpy as np
import os
import sys
from typing import List, Tuple
import statistics

from mythologizer_postgres.connectors.myth_store import insert_myth, insert_myths_bulk
from mythologizer_postgres.db import clear_all_rows, get_table_row_counts


def generate_test_data(count: int, embedding_dim: int = 4) -> Tuple[List[np.ndarray], List[List[int]], List[List[np.ndarray]], List[List[float]]]:
    """
    Generate test data for benchmarking.
    
    Args:
        count: Number of myths to generate
        embedding_dim: Dimension of embeddings
    
    Returns:
        Tuple of (main_embeddings, embedding_ids_list, offsets_list, weights_list)
    """
    main_embeddings = []
    embedding_ids_list = []
    offsets_list = []
    weights_list = []
    
    for i in range(count):
        # Generate random main embedding
        main_embedding = np.random.randn(embedding_dim).astype(np.float32)
        main_embeddings.append(main_embedding)
        
        # Generate 2-5 nested embeddings per myth
        num_nested = np.random.randint(2, 6)
        
        # Generate embedding IDs (1-1000)
        embedding_ids = np.random.randint(1, 1001, num_nested).tolist()
        embedding_ids_list.append(embedding_ids)
        
        # Generate offset vectors
        offsets = [np.random.randn(embedding_dim).astype(np.float32) for _ in range(num_nested)]
        offsets_list.append(offsets)
        
        # Generate weights (sum to 1.0)
        weights = np.random.rand(num_nested).astype(np.float32)
        weights = (weights / weights.sum()).tolist()
        weights_list.append(weights)
    
    return main_embeddings, embedding_ids_list, offsets_list, weights_list


def benchmark_single_insertions(data: Tuple[List[np.ndarray], List[List[int]], List[List[np.ndarray]], List[List[float]]]) -> float:
    """
    Benchmark single myth insertions.
    
    Args:
        data: Tuple of (main_embeddings, embedding_ids_list, offsets_list, weights_list)
    
    Returns:
        Total time in seconds
    """
    main_embeddings, embedding_ids_list, offsets_list, weights_list = data
    
    start_time = time.time()
    
    for i in range(len(main_embeddings)):
        insert_myth(
            main_embedding=main_embeddings[i],
            embedding_ids=embedding_ids_list[i],
            offsets=offsets_list[i],
            weights=weights_list[i]
        )
    
    end_time = time.time()
    return end_time - start_time


def benchmark_bulk_insertions(data: Tuple[List[np.ndarray], List[List[int]], List[List[np.ndarray]], List[List[float]]]) -> float:
    """
    Benchmark bulk myth insertions.
    
    Args:
        data: Tuple of (main_embeddings, embedding_ids_list, offsets_list, weights_list)
    
    Returns:
        Total time in seconds
    """
    main_embeddings, embedding_ids_list, offsets_list, weights_list = data
    
    start_time = time.time()
    
    insert_myths_bulk(
        main_embeddings=main_embeddings,
        embedding_ids_list=embedding_ids_list,
        offsets_list=offsets_list,
        weights_list=weights_list
    )
    
    end_time = time.time()
    return end_time - start_time


def run_benchmark(count: int, embedding_dim: int = 4, num_runs: int = 3) -> Tuple[List[float], List[float]]:
    """
    Run benchmark for a specific count multiple times.
    
    Args:
        count: Number of myths to insert
        embedding_dim: Dimension of embeddings
        num_runs: Number of benchmark runs
    
    Returns:
        Tuple of (single_times, bulk_times)
    """
    single_times = []
    bulk_times = []
    
    print(f"Running benchmark for {count} myths ({num_runs} runs)...")
    
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}")
        
        # Clear database before each run
        clear_all_rows()
        
        # Generate fresh test data
        data = generate_test_data(count, embedding_dim)
        
        # Benchmark single insertions
        single_time = benchmark_single_insertions(data)
        single_times.append(single_time)
        
        # Clear database
        clear_all_rows()
        
        # Benchmark bulk insertions
        bulk_time = benchmark_bulk_insertions(data)
        bulk_times.append(bulk_time)
        
        print(f"    Single: {single_time:.4f}s, Bulk: {bulk_time:.4f}s")
    
    return single_times, bulk_times


def print_results(counts: List[int], results: List[Tuple[List[float], List[float]]]):
    """
    Print formatted benchmark results.
    
    Args:
        counts: List of myth counts tested
        results: List of (single_times, bulk_times) for each count
    """
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"{'Count':<8} {'Single (s)':<15} {'Bulk (s)':<15} {'Speedup':<12} {'Single/rec':<15} {'Bulk/rec':<15}")
    print("-"*80)
    
    for count, (single_times, bulk_times) in zip(counts, results):
        single_mean = statistics.mean(single_times)
        bulk_mean = statistics.mean(bulk_times)
        speedup = single_mean / bulk_mean if bulk_mean > 0 else float('inf')
        
        single_per_record = single_mean / count
        bulk_per_record = bulk_mean / count
        
        print(f"{count:<8} {single_mean:<15.4f} {bulk_mean:<15.4f} {speedup:<12.2f}x {single_per_record:<15.6f} {bulk_per_record:<15.6f}")
    
    print("-"*80)
    print("Legend:")
    print("  Count: Number of myths inserted in each test")
    print("  Single (s): Total time to insert all myths one by one (average of 3 runs)")
    print("  Bulk (s): Total time to insert all myths in bulk (average of 3 runs)")
    print("  Speedup: How many times faster bulk insertion is compared to single insertion")
    print("  Single/rec: Average time per individual myth insertion")
    print("  Bulk/rec: Average time per myth when using bulk insertion")


def main():
    """Main benchmarking function."""
    # Get embedding dimension from environment or default to 4
    embedding_dim = int(os.getenv('EMBEDDING_DIM', '4'))
    
    # Test counts
    counts = [10, 20, 50, 100, 200]
    
    print(f"Starting myth insertion benchmark")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Test counts: {counts}")
    print(f"Runs per test: 3")
    print()
    
    # Check if database is available
    try:
        from mythologizer_postgres.db import ping_db
        if not ping_db():
            print("ERROR: Database is not available. Please ensure the test database is running.")
            print("Run: make fresh")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to connect to database: {e}")
        print("Please ensure the test database is running.")
        print("Run: make fresh")
        sys.exit(1)
    
    # Clear database before starting
    try:
        clear_all_rows()
        print("Database cleared for benchmarking.")
    except Exception as e:
        print(f"Warning: Could not clear database: {e}")
    
    # Run benchmarks
    results = []
    for count in counts:
        single_times, bulk_times = run_benchmark(count, embedding_dim, num_runs=3)
        results.append((single_times, bulk_times))
    
    # Print results
    print_results(counts, results)
    
    # Print summary statistics
    print("\nSUMMARY:")
    print("-"*40)
    
    all_single_times = [time for result in results for time in result[0]]
    all_bulk_times = [time for result in results for time in result[1]]
    
    print(f"Total single insertion time: {sum(all_single_times):.4f}s")
    print(f"Total bulk insertion time: {sum(all_bulk_times):.4f}s")
    print(f"Overall speedup: {sum(all_single_times) / sum(all_bulk_times):.2f}x")
    
    # Clean up
    try:
        clear_all_rows()
        print("\nDatabase cleaned up.")
    except Exception as e:
        print(f"\nWarning: Could not clean up database: {e}")


if __name__ == "__main__":
    main()
