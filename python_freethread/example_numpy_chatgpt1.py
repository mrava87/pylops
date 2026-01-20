# save as numpy_free_threaded_benchmark.py
import time
import numpy as np
import pprint
from concurrent.futures import ThreadPoolExecutor
from threadpoolctl import threadpool_info, threadpool_limits


# -------------------------------
# Benchmark functions
# -------------------------------
def add_arrays(n: int) -> None:
    """Simple elementwise addition (single-threaded inside each call)."""
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    return A + B


def add_arrays_serial(n: int, n_runs: int = 4) -> None:
    """Run elementwise addition sequentially (no Python threads)."""
    for _ in range(n_runs):
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        _ = A + B


def matmul_arrays(n: int) -> None:
    """Matrix multiplication (uses BLAS)."""
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    # Limit BLAS to 1 thread per Python thread
    with threadpool_limits(limits=1):
        return A @ B


def matmul_arrays_serial(n: int, n_runs: int = 4) -> None:
    """Run matrix multiplication sequentially (no Python threads)."""
    for _ in range(n_runs):
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        with threadpool_limits(limits=4):
            _ = A @ B


# -------------------------------
# Helper: run threaded benchmark
# -------------------------------
def run_serial(func, n):
    start = time.perf_counter()
    func(n)
    return time.perf_counter() - start


def run_threaded(func, n: int, n_threads: int, n_runs: int = 4) -> float:
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        list(pool.map(func, [n] * n_runs))
    return time.perf_counter() - start


# -------------------------------
# Main benchmark
# -------------------------------
def main():
    Nadd = 10_000  # Matrix size for add (adjust as needed)
    Nmm = 5_000  # Matrix size for matmul (adjust as needed)

    # # --- Non-threaded baseline for A + B ---
    # print("\n--- add_arrays (sequential baseline) ---")
    # elapsed = run_serial(add_arrays_serial, Nadd)
    # print(f" serial (1 runs): {elapsed / 4.:6.2f} s")
    # print(f" serial (2 sequential runs): {elapsed / 2.:6.2f} s")
    # print(f" serial (4 sequential runs): {elapsed:6.2f} s")

    # # --- Threaded A + B ---
    # print("\n--- add_arrays (parallel with ThreadPoolExecutor) ---")
    # for threads in (1, 2, 4):
    #     elapsed = run_threaded(add_arrays, Nadd, threads, 4)
    #     print(f"{threads:>2} threads ({4:>2} run): {elapsed:6.2f} s")

    # --- Non-threaded baseline for A @ B ---
    print("\n--- matmul_arrays (sequential baseline) ---")
    elapsed = run_serial(matmul_arrays_serial, Nmm)
    print(f" serial (1 runs): {elapsed / 4.:6.2f} s")
    print(f" serial (2 sequential runs): {elapsed / 2.:6.2f} s")
    print(f" serial (4 sequential runs): {elapsed:6.2f} s")

    # --- Threaded A @ B ---
    print("\n--- matmul_arrays (parallel with ThreadPoolExecutor) ---")
    for threads in (1, 2, 4):
        elapsed = run_threaded(matmul_arrays, Nmm, threads, 4)
        print(f"{threads:>2} threads ({4:>2} run): {elapsed:6.2f} s")


if __name__ == "__main__":
    pprint.pp(threadpool_info())
    main()
