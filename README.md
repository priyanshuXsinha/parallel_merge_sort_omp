# Parallel Merge Sort — ME522 HPSC
**IIT Mandi · Prof. Gaurav Bhutani · 2025**

A parallel implementation of merge sort using **OpenMP** in C++, demonstrating fork-join parallelism, thread synchronisation, and Amdahl's Law.

---

## Results

| Metric | Value |
|---|---|
| Array Size | 1,000,000 integers |
| Serial Time | 0.1475 s |
| Parallel Time (4 threads) | 0.0561 s |
| **Speedup** | **2.63×** |
| Parallel Efficiency | 65.7% |
| Amdahl Prediction | 2.76× (f = 0.85) |

---

## Build & Run

```bash
# Compile
g++ -O2 -fopenmp parallel_mergesort.cpp -o sort

# Run (default: N=1,000,000, 4 threads)
./sort

# Custom: 2M elements, 8 threads
./sort 2000000 8
```

---

## How It Works

**Phase 1 — Parallel Sort**
The array is split into `p` equal chunks. Each OpenMP thread independently sorts its chunk using serial merge sort. Since chunks are disjoint, there are no data races — no mutex or `#pragma omp critical` needed inside the sort.

```cpp
#pragma omp parallel num_threads(nthreads)
{
    int tid  = omp_get_thread_num();
    int from = tid * chunk;
    int to   = std::min(from + chunk - 1, n - 1);
    mergeSort(arr, from, to);   // no data race
}
// implicit barrier — all threads sync here
```

**Phase 2 — Serial Merge**
After the barrier, the master thread merges all sorted chunks. This is the Amdahl bottleneck (~15% of total work).

---

## Project Structure

```
.
├── parallel_mergesort.cpp   # main source
├── ME522_report.tex         # LaTeX report
└── README.md
```

---

## ME522 Concepts Covered

- OpenMP fork-join model (`#pragma omp parallel`)
- Implicit thread barrier and synchronisation
- `omp_get_wtime()` profiling
- Amdahl's Law and parallel efficiency
- Compiler optimisation flags (`-O2 -fopenmp`)

---

## Requirements

- GCC / Clang with OpenMP support
- Linux / macOS (tested on Apple M-series, macOS)
