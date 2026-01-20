from concurrent.futures import ThreadPoolExecutor
from threadpoolctl import threadpool_limits
import numpy as np
import time

def heavy_compute(n, limits=1):
    with threadpool_limits(limits=limits):  # limit BLAS to 1 thread per Python thread
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        return A + B

def main():
    N = 10000
    start = time.perf_counter()
    # for _ in range(4):
    #     heavy_compute(N, limits=4)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(heavy_compute, [N] * 4))
    print(f"Total time: {time.perf_counter() - start:.2f}s")

if __name__ == "__main__":
    main()
