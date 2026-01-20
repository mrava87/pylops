import sys
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor


def count(n):
    total = 0
    for i in range(n):
        total += i
    return total

def worker(n):
    t0 = time.perf_counter()
    tot = count(n)
    print(f"Done in {time.perf_counter() - t0:.2f}s")
    return tot

def main(multithreaded):
    N = 200_000_000  # big number to make CPU-bound
    workers: int = 8  # no of CPU cores to use
    
    t0 = time.perf_counter()
    if multithreaded:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # for _ in range(workers):
            #     executor.submit(lambda: worker(N))
            y = list(executor.map(lambda N: worker(N), [N]*workers))
    else:
        for _ in range(workers):
            worker(N)
    tend = time.perf_counter()
    print(f"Total wall time: {tend - t0:.2f}s")
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
