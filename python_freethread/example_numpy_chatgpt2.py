import numpy as np, time, sys
from concurrent.futures import ThreadPoolExecutor

def add_arrays(n=4000):
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    C = A + B
    for _ in range(5):
        C = (np.sin(C) + np.exp(C)) / 1e4   # compute-heavy
    return C.sum()

def run_sequential(tasks=8, n=4000):
    start = time.perf_counter()
    for _ in range(tasks):
        add_arrays(n)
    print(f"Sequential: {time.perf_counter()-start:.2f}s")

def run_parallel(tasks=8, n=4000, workers=4):
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda _: add_arrays(n), range(tasks)))
    print(f"Parallel({workers}): {time.perf_counter()-start:.2f}s")

if __name__ == "__main__":
    print("Use GIL:", sys._is_gil_enabled())
    run_sequential(tasks=8, n=4000)
    run_parallel(tasks=8, n=4000, workers=4)

