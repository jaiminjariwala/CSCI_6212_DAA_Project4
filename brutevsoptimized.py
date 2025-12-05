import time
import random
import math
import matplotlib.pyplot as plt

# -----------------------------
#  Naive DP ~ O(Δ n^3)
# -----------------------------
def naive_dp(n, delta):
    """
    Simulate a naive dynamic program whose time
    complexity is proportional to O(delta * n^3).

    We just do arithmetic work in nested loops so that
    the runtime scales like the theory predicts.
    """
    total = 0
    for _ in range(delta):
        for i in range(n):
            for j in range(n):
                # inner loop is scaled down (n//10) so it doesn't get too slow
                for k in range(max(1, n // 10)):
                    total += (i + j + k) % 7
    return total


# -----------------------------
#  Optimized DP ~ O(Δ n^2 log n)
# -----------------------------
def optimized_dp(n, delta):
    """
    Simulate an optimized algorithm whose complexity is
    O(delta * n^2 log n). We do n sorts of size n inside
    an outer loop over delta and i.
    """
    total = 0
    for _ in range(delta):
        for i in range(n):
            # create an array and sort it: O(n log n)
            arr = [random.random() for _ in range(n)]
            arr.sort()
            # do a little extra O(n) work over a subset
            step = max(1, n // 10)
            for val in arr[::step]:
                total += val > 0.5
    return total


# -----------------------------
#  Experiment runner
# -----------------------------
def run_experiment(ns, delta):
    naive_times = []
    opt_times = []

    for n in ns:
        print(f"Running for n = {n} ...")

        start = time.perf_counter()
        naive_dp(n, delta)
        naive_elapsed = time.perf_counter() - start
        naive_times.append(naive_elapsed)

        start = time.perf_counter()
        optimized_dp(n, delta)
        opt_elapsed = time.perf_counter() - start
        opt_times.append(opt_elapsed)

        print(f"  Naive   : {naive_elapsed:.4f} s")
        print(f"  Optimized: {opt_elapsed:.4f} s")

    return naive_times, opt_times


def main():
    # Sizes of the graph (number of vertices)
    ns = [20, 30, 40, 50, 60]   # you can increase these if your machine is fast
    delta = 3                   # number of allowed refueling stops

    naive_times, opt_times = run_experiment(ns, delta)

    # -----------------------------
    #  Plot the results
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(ns, naive_times, marker='o', label='Naive DP  ~ O(Δ n³)')
    plt.plot(ns, opt_times, marker='o', label='Optimized DP ~ O(Δ n² log n)')

    plt.xlabel('Number of vertices n')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime Comparison: Naive vs Optimized Algorithm\n(Gas Station Problem – Simulated Work)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print the raw data in a nice table for your slide
    print("\nExperimental runtimes:")
    print(" n   naive_time (s)   optimized_time (s)")
    for n, t1, t2 in zip(ns, naive_times, opt_times):
        print(f"{n:2d}   {t1:>10.4f}        {t2:>10.4f}")


if __name__ == "__main__":
    main()
