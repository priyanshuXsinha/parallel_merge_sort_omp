// ============================================================
//  Parallel Merge Sort — ME522 HPSC Project
//  IIT Mandi · Gaurav Bhutani
//
//  Compile:
//    g++ -O2 -fopenmp parallel_mergesort.cpp -o sort
//
//  Run:
//    ./sort              (uses default N=1,000,000, threads=4)
//    ./sort 2000000 8    (custom N and thread count)
// ============================================================

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <omp.h>

// ─── Serial Merge ────────────────────────────────────────────
void merge(std::vector<int>& arr, int left, int mid, int right) {
    std::vector<int> L(arr.begin() + left,  arr.begin() + mid + 1);
    std::vector<int> R(arr.begin() + mid + 1, arr.begin() + right + 1);

    int i = 0, j = 0, k = left;
    while (i < (int)L.size() && j < (int)R.size())
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < (int)L.size()) arr[k++] = L[i++];
    while (j < (int)R.size()) arr[k++] = R[j++];
}

// ─── Serial Merge Sort ───────────────────────────────────────
void mergeSort(std::vector<int>& arr, int left, int right) {
    if (left >= right) return;
    int mid = (left + right) / 2;
    mergeSort(arr, left, mid);
    mergeSort(arr, mid + 1, right);
    merge(arr, left, mid, right);
}

// ─── Parallel Merge Sort (OpenMP) ───────────────────────────
//   Strategy:
//   1. Divide array into 'nthreads' chunks
//   2. Each thread sorts its chunk independently (parallel)
//   3. After barrier, merge all chunks serially
//   This maps directly to the fork-join model from lectures.
void parallelMergeSort(std::vector<int>& arr, int nthreads) {
    int n = arr.size();
    int chunk = (n + nthreads - 1) / nthreads;  // ceil division

    // ── Phase 1: Parallel sort each chunk ──────────────────
    #pragma omp parallel num_threads(nthreads)
    {
        int tid  = omp_get_thread_num();
        int from = tid * chunk;
        int to   = std::min(from + chunk - 1, n - 1);

        if (from < n) {
            #pragma omp critical
            std::cout << "  [Thread " << tid << "] sorting indices ["
                      << from << ".." << to << "]\n";

            mergeSort(arr, from, to);

            #pragma omp critical
            std::cout << "  [Thread " << tid << "] done.\n";
        }
    }
    // implicit barrier at end of parallel region

    // ── Phase 2: Serial merge of sorted chunks ─────────────
    // (simulates the reduction/gather step in MPI too)
    std::cout << "  [Main ] All threads synced. Merging chunks...\n";

    int width = chunk;
    while (width < n) {
        for (int i = 0; i < n; i += 2 * width) {
            int left  = i;
            int mid   = std::min(i + width - 1, n - 1);
            int right = std::min(i + 2 * width - 1, n - 1);
            if (mid < right)
                merge(arr, left, mid, right);
        }
        width *= 2;
    }
}

// ─── Verify sorted ──────────────────────────────────────────
bool isSorted(const std::vector<int>& arr) {
    for (int i = 1; i < (int)arr.size(); i++)
        if (arr[i] < arr[i-1]) return false;
    return true;
}

// ─── Print Amdahl speedup table ─────────────────────────────
void printAmdahl(double serialTime, double parallelTime, int nthreads) {
    double speedup    = serialTime / parallelTime;
    double efficiency = speedup / nthreads * 100.0;
    // Amdahl's law: assume ~85% of work is parallelisable
    double fpar = 0.85;
    double theoretical = 1.0 / ((1 - fpar) + fpar / nthreads);

    std::cout << "\n╔══════════════════════════════════════════╗\n";
    std::cout <<   "║         PERFORMANCE ANALYSIS             ║\n";
    std::cout <<   "╠══════════════════════════════════════════╣\n";
    std::cout <<   "║ Threads        : " << std::setw(5) << nthreads     << "                    ║\n";
    std::cout <<   "║ Serial time    : " << std::setw(8) << std::fixed << std::setprecision(4) << serialTime   << " s              ║\n";
    std::cout <<   "║ Parallel time  : " << std::setw(8) << parallelTime << " s              ║\n";
    std::cout <<   "║ Measured speedup: " << std::setw(6) << std::setprecision(2) << speedup   << "x               ║\n";
    std::cout <<   "║ Amdahl speedup : " << std::setw(6) << theoretical  << "x (f=0.85)      ║\n";
    std::cout <<   "║ Efficiency     : " << std::setw(5) << std::setprecision(1) << efficiency << "%                  ║\n";
    std::cout <<   "╚══════════════════════════════════════════╝\n";
}

// ─── Main ────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    int N        = (argc > 1) ? std::atoi(argv[1]) : 1000000;
    int nthreads = (argc > 2) ? std::atoi(argv[2]) : 4;

    std::cout << "================================================\n";
    std::cout << " Parallel Merge Sort — ME522 HPSC\n";
    std::cout << " N = " << N << "  |  Threads = " << nthreads << "\n";
    std::cout << "================================================\n\n";

    // Generate random data
    srand(42);
    std::vector<int> original(N);
    for (int i = 0; i < N; i++) original[i] = rand() % 1000000;

    // ── Serial run ──
    std::vector<int> arrSerial = original;
    std::cout << "[1/2] Running serial sort...\n";
    double t0 = omp_get_wtime();
    mergeSort(arrSerial, 0, N - 1);
    double serialTime = omp_get_wtime() - t0;
    std::cout << "      Done. Sorted: " << (isSorted(arrSerial) ? "YES ✓" : "NO ✗") << "\n\n";

    // ── Parallel run ──
    std::vector<int> arrPar = original;
    std::cout << "[2/2] Running parallel sort (" << nthreads << " threads)...\n";
    double t1 = omp_get_wtime();
    parallelMergeSort(arrPar, nthreads);
    double parallelTime = omp_get_wtime() - t1;
    std::cout << "      Done. Sorted: " << (isSorted(arrPar) ? "YES ✓" : "NO ✗") << "\n";

    // ── Results ──
    printAmdahl(serialTime, parallelTime, nthreads);

    // ── Speedup sweep (bonus: try 1,2,4,8 threads) ──────────
    std::cout << "\n── Speedup Sweep (Amdahl's Law, f_par=0.85) ─\n";
    std::cout << std::left << std::setw(12) << "Threads"
              << std::setw(16) << "Theoretical"
              << "Bar\n";
    std::cout << std::string(46, '-') << "\n";
    for (int p : {1, 2, 4, 8, 16}) {
        double sp = 1.0 / (0.15 + 0.85 / p);
        int bars = (int)(sp / 5.5 * 30);
        std::cout << std::setw(12) << p
                  << std::setw(16) << std::fixed << std::setprecision(2) << sp
                  << std::string(bars, '█') << "\n";
    }

    return 0;
}
