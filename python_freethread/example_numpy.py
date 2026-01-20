import sys
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from threadpoolctl import threadpool_limits
import numpy as np


def count(n):
    total = 0
    for i in range(n):
        total += i
    return total

def worker(n):
    with threadpool_limits(limits=1):
        t0 = time.perf_counter()
        tot = np.dot(np.ones(n), np.ones(n))
        print(f"Done in {time.perf_counter() - t0:.2f}s")
        return tot

def main(multithreaded):
    N = 200_000_000  # big number to make CPU-bound
    Nrepeat = 40
    workers: int = 4  # no of CPU cores to use
    
    t0 = time.perf_counter()
    if multithreaded:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # for _ in range(Nrepeat):
            #     executor.submit(lambda: worker(N))
            y = list(executor.map(lambda N: worker(N), [N] * Nrepeat))
    else:
        y = []
        for _ in range(Nrepeat):
            y.append(worker(N))
    tend = time.perf_counter()
    print(f"Total wall time: {tend - t0:.2f}s")
    print(y)
    print("Done.")


if __name__ == "__main__":
    print("Use GIL:", sys._is_gil_enabled())
    parser = ArgumentParser(description="Scrape Hacker News stories and comments.")
    parser.add_argument(
        "--multithreaded",
        action="store_true",
        default=False,
        help="Use multithreading.",
    )
    args = parser.parse_args()
    main(args.multithreaded)
