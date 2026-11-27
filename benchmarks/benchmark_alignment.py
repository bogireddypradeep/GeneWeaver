
"""
GeneWeaver - Smith-Waterman Performance Benchmark

Measures optimized CPU alignment performance using Numba.
"""

import time
import numpy as np

from src.alignment import smith_waterman


def run_benchmark(sequence_length=500, repetitions=5):
    """Benchmark Smith-Waterman alignment."""

    query = "ACGT" * (sequence_length // 4)
    reference = "TGCA" * (sequence_length // 4)

    # Include remaining bases if sequence length is not divisible by 4.
    query = (query + "ACGT")[:sequence_length]
    reference = (reference + "TGCA")[:sequence_length]

    print("=" * 55)
    print("GENEWEAVER - ALIGNMENT PERFORMANCE BENCHMARK")
    print("=" * 55)

    print(f"Sequence length: {sequence_length} bp")
    print(f"Matrix dimensions: {sequence_length + 1} x {sequence_length + 1}")

    # Warm-up: compile the Numba kernel before measuring.
    print("\nWarming up Numba JIT...")
    result = smith_waterman(query, reference)

    print("Warm-up completed.")
    print(f"Alignment score: {result.score}")

    # Measure repeated executions.
    execution_times = []

    print("\nRunning benchmark...")

    for run in range(repetitions):
        start = time.perf_counter()

        smith_waterman(query, reference)

        elapsed = time.perf_counter() - start
        execution_times.append(elapsed)

        print(f"Run {run + 1}: {elapsed:.6f} seconds")

    average_time = sum(execution_times) / len(execution_times)
    best_time = min(execution_times)
    worst_time = max(execution_times)

    # Memory estimate for the optimized DP arrays.
    rows = sequence_length + 1
    cols = sequence_length + 1

    traceback_memory = rows * cols * np.dtype(np.uint8).itemsize
    score_rows_memory = 2 * cols * np.dtype(np.int32).itemsize
    total_memory = traceback_memory + score_rows_memory

    # Original implementation: full int32 score matrix + uint8 traceback.
    original_memory = (
        rows * cols * np.dtype(np.int32).itemsize
        + rows * cols * np.dtype(np.uint8).itemsize
    )

    print("\n" + "=" * 55)
    print("BENCHMARK RESULTS")
    print("=" * 55)

    print(f"Average time: {average_time:.6f} seconds")
    print(f"Best time:    {best_time:.6f} seconds")
    print(f"Worst time:   {worst_time:.6f} seconds")

    print("\nMEMORY ESTIMATE")
    print(f"Original DP arrays:  {original_memory / 1024:.2f} KiB")
    print(f"Optimized DP arrays: {total_memory / 1024:.2f} KiB")

    print("\n" + "=" * 55)


if __name__ == "__main__":
    run_benchmark()
