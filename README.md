# fastnum

`fastnum` is a small, header-only C++ library providing **numerically stable, online (streaming) statistics** with **constant memory usage** and **mergeable state**.

It is designed for:
- streaming data,
- large datasets that cannot be stored in memory,
- parallel / partitioned computation where partial results must be merged.

All components are tested for correctness and merge equivalence.

---

## Features

### Implemented (v1.0)

- **RunningStats**
  - Online mean and variance (population & sample)
  - Numerically stable (Welford)
  - Mergeable across partitions

- **OnlineStandardScaler**
  - Streaming z-score standardization
  - Readiness-aware (`ready()` gating)
  - Mergeable for parallel fitting

- **OnlineCovariance**
  - Online covariance and correlation
  - Tracks per-dimension variance
  - Mergeable with exact equivalence to single-pass computation

All algorithms operate in **O(1) memory** and **O(1) time per observation**.

---

## Design principles

- **Online / streaming first**  
  All statistics are updated incrementally as samples arrive.

- **Mergeable state**  
  Any accumulator can be merged with another instance to produce the same result
  as observing the combined data in a single pass (up to floating-point roundoff).

- **Explicit readiness and NaN policy**  
  Statistics that are undefined (e.g. variance with fewer than 2 samples) return `NaN`.
  Components expose a `ready()` method to explicitly signal when results are meaningful.

- **Header-only, no allocations**  
  All algorithms are implemented in headers and do not allocate memory internally.

---

## NaN and readiness policy

- Undefined quantities (e.g. variance with insufficient samples) return `NaN`
- `ready()` indicates whether an object can produce meaningful results
- Transform operations return or fill `NaN` when not ready
- Input `NaN`s propagate naturally through the computations

This explicit policy avoids silent failures and makes downstream issues easy to detect.

---

## Usage examples

### RunningStats

```cpp
#include <fastnum/running_stats.hpp>

fastnum::RunningStats<double> rs;

rs.observe(1.0);
rs.observe(2.0);
rs.observe(3.0);

double mean = rs.mean();
double var  = rs.variance_sample();
```
#### Merging partial results:
```cpp
fastnum::RunningStats<double> a, b;

// observe data into a and b independently
a.observe(1.0);
a.observe(2.0);

b.observe(3.0);
b.observe(4.0);

// combine
a.merge(b);

double mean = a.mean();
```

### OnlineStandardScaler
```cpp
#include <fastnum/online_standard_scaler.hpp>

fastnum::OnlineStandardScaler<double> scaler;

scaler.observe(1.0);
scaler.observe(2.0);
scaler.observe(3.0);

if (scaler.ready()) {
    double z = scaler.transform(2.5);
}
```
#### Batch Usage:
```cpp
#include <vector>

std::vector<double> data = {1.0, 2.0, 3.0, 4.0};

fastnum::OnlineStandardScaler<double> scaler;
scaler.observe(data);

// Standardize in place
scaler.transform_inplace(data);
```

### OnlineCovariance
```cpp
#include <fastnum/online_covariance.hpp>

fastnum::OnlineCovariance<double> cov;

cov.observe(1.0, 2.0);
cov.observe(2.0, 3.0);
cov.observe(3.0, 5.0);

if (cov.ready()) {
    double c = cov.covariance_sample();
    double r = cov.correlation();
}
```
#### Merging partial results:
```cpp
fastnum::OnlineCovariance<double> a, b;

// observe paired data into a and b independently
a.observe(1.0, 2.0);
a.observe(2.0, 3.0);

b.observe(3.0, 5.0);
b.observe(4.0, 8.0);

// combine
a.merge(b);

double corr = a.correlation();
```

## Testing

All components are tested using **Catch2**, with a focus on correctness and composability:

- correctness checks against naive reference implementations
- merge equivalence tests (split vs. single-pass)
- readiness and NaN behavior validation
- stream vs. batch equivalence

All tests pass under standard configurations.

---

## Build

`fastnum` is header-only and requires a **C++20-compatible compiler**.

To build tests and examples:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

You can disable tests or examples via CMake options:

```bash
cmake -DFASTNUM_BUILD_TESTS=OFF -DFASTNUM_BUILD_EXAMPLES=OFF ..
```

---

## Non-goals

The following are intentionally out of scope for v1.0:

- Thread safety (external synchronization required)
- SIMD/vectorized batch algorithms
- Quantiles, sketches, or histogram-based statistics

These may be considered future work.

---

## License

MIT


