
# OnlineCovariance Performance Investigation

*(Micro Benchmarking -> Bottleneck Identification -> Optimization Direction)*

## 1. Goal

- The goal of this investigation was to understand the performance characteristics of the OnlineCovariance::observe kernel and determine **whether optimization is warranted**, and if so, what class of optimization is likely to matter

	We explicitly wanted to avoid blind optimization


## 2. Kernel Under Study

The core operation is [Welford-style online covariance](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance):
- Per-element update 
- Strong loop-carried dependences (means and second moments)
- Exact numerical correctness (no fast-math assumptions)

Two usage patterns were considered:
1. Scalar streaming: repeated calls to observe(x,y)
2. Array streaming: observe(xs, ys, N)



## 3. Benchmarking Methodology

### Tooling 
- Google Benchmark for stable, repeatable microbenchmarks
- CPU affinity pinned (taskset -c)
- Release builds (-O3, -DNDEBUG)
- Benchmarks run long enough to stabilize (low CV)

### Benchmarks Implemented 

#### Scalar steady-state benchmark:
	Purpose 
		- Measure the minimum per-update cost of observe (x,y)
		- Establish a lower bound

	Key Properties
		- Warm-up to steady state (n_ >= 1000)
		- Single accumulator reused
		- Compiler prevented from eliding work

	Result:
		- ~2.5ns per update
		- Confirms very tight inner loop

#### Array Streaming Benchmark
	Purpose
		- Measure throughput on realistic contiguous input
		- Determine whether the kernel is memory-bound or compute-bound

	Key design choices
		- xs and ys allocated and filled once (outside timing)
		- Streaming access pattern
		- Lower-bound bandwidth counter added

Lower-bound bandwidth Calc:
		
```cpp
	bytes_min = 2 * N * sizeof(double)
```


## 4. Benchmark Results summary 

#### Throughput
- ~250 million elements/second
- Throughput flat across N (1k -> 1M)

#### Effective Input Bandwidth (lower bound)
- Increases with N 
- Plateaus far below DRAM bandwidth

#### Interpretation
- Kernel is not **memory-bounded**
- Performance dominated by per-element computation and dependencies
- Cache and memory optimizations are unlikely to help


## 5. Conceptual Insight: "Pseudo Arithmetic Identity"

Although we did not compute formal roofline arithmetic intensity (FLOPs / DRAM bytes), the lower-bound bandwidth metric serves as a proxy:

- If effective bandwidth $\approx$ hardware bandwidth -> memory-bound
- If effective bandwidth < hardware bandwidth -> compute or dependency-bound

This placed the kernel firmly in the compute / latency-bound regime

## 6. Instruction-Level Investigation (GodBolt)

Because perf was unavailable on WSL, Compiler Explorer (Godbolt) was used to inspect generated assembly for the hot loop.

### Key Observations
Inside loop:

```asm
	cvtsi2sd   xmm3, rax    ; int → double
	divsd      xmm5, xmm3   ; compute 1 / n
```
- One scaler divide per element
- Integer -> FP conversion per iteration
- All subsequent computations depend on the result


The compiler:
- Fully inlined the loop
- Kept state in registers
- Introduced no unnecessary spills

Conclusion:
- The kernel is latency-bound on a long dependency chain, dominated by scalar division.


## 7. What We Learned (Negative Results Also Matter)

### Optimizations that are not worth pursuing:
- Cache blocking
- Prefetching
- Memory layout changes
- NUMA tuning


## 8. Promising Optimization Directions

Only transformations that reduce dependency depth or divide frequency are likely to help

### 8.1 Multi-accumulator blocking
- Maintain K independent accumulators
- Process data in round-robin or blocks
- Merge accumulators at the end

Benefits:
- Breaks dependency chain
- Increases instruction-level parallelism (ILP
- Preserves correctness via associative merge



### 8.2 Block-wise Welford (algorithmic reshape)
- Accumulate per-block stats
- Merge blocks 
- Reduces divides from O(N) -> O(N / block size)
Higher complexity, higher potential payoff


## 9. Validation: K-Accumulators Implementation & Results
### Implementation Summary
A K-accumulator variant of the array streaming path was implemented:
- Input stream partitioned across $K = 4$ **independent OnlineCovariance accumulators**
- Each accumulator processed every K-th element (round robin)
- Partial statistics merged at the end using Welford merge formulas
- Merge implementation reduced divides from four to one per merge
- No fast-math assumptions introduced

### Benchmark Setup (Controlled Comparison)
- Same benchmark harness and inputs as baseline array streaming test
- CPU pinned, release build ( -O3, -DNDEBUG)
- Identical warm-up procedure
- Benchmarks run with MinTime(0.5) to reduce noise
- Measured metric: sustained elements/sec

### Results

| N   | Baseline Throughput | K = 4 Throughput | Speedup |
| --- | ------------------- | ---------------- | ------- |
| 1k  | ~340 M elem/s       | ~395-400 M/s     | ~15-18% |
| 64k | ~360 M elem/s       | ~425 M/s         | ~18%    |
| 1M  | ~345 M elem/s       | ~405 M/s         | ~17-20% |
Results were stable across runs ($CV \approx 2\%$).

### Interpretation
- The K-accumulator approach consistently improved throughput by ~15-20%
- This confirms the original hypothesis that the kernel is **latency / dependency-chain bound**
- Breaking the loop-carried dependency exposed additional instruction-level parallelism
- Further manual loop unrolling did **not** improve performance and slightly regressed it, likely due to increased register pressure
This indicates the optimized kernel is close to the core's throughput limits for this instruction mix.

### Negative Results
- Manual loop unrolling did not improve throughput
- Increasing unrolling increased variance and reduced performance
- Confirms diminishing returns once sufficient ILP is exposed
## 10. Key Takeaways
- Microbenchmarks are for classification, optimization
- Assembly is a diagnostic tool, not something to master
- Lower-bound metrics are powerful even when approx
- Avoiding blind optimizations saved significant time 
- Dependency-breaking transformations can yield meaningful gains in tight numeric kernels
- K-accumulator blocking is effective up to a small K; beyond that, register pressure dominates

Most importantly:
- We now know why the kernel is slow before trying to make it faster

## Next Steps 

1. Generalize K-accumulator path with compile-time K (2/4/8) and select best default
2. Add numerical validation tests vs baseline (relative error bounds)
3. Document reproducibility trade-offs (bitwise vs. statistical equivalence)
4. Stop -> Further micro optimizations unlikely to pay off

## Meta Lesson 

This process followed the correct performance engineering loop:

1. Measure 
2. Bound
3. Inspect
4. Hypothesize
5. Test one change
That loop matters more than any single optimization