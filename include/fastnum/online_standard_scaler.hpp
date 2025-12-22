#pragma once
#include <cstddef>
#include <limits>
#include <type_traits>
#include <cmath>
#include <fastnum/running_stats.hpp>

namespace fastnum {

/**
 * @brief Online (streaming) standardization using running mean/variance.
 *
 * `OnlineStandardScaler` implements the classic standard score (z-score)
 * transform:
 *
 * \f[
 *   z = \frac{x - \mu}{\sigma}
 * \f]
 *
 * where \f$\mu\f$ and \f$\sigma\f$ are estimated incrementally from incoming
 * samples. This is useful when:
 * - data arrives in a stream,
 * - you want constant-memory fitting,
 * - you need to fit on large datasets without storing them,
 * - you want to merge partial fits across threads/partitions.
 *
 * ## Requirements / Assumptions
 * - `T` must be a floating-point type (`float`, `double`, `long double`).
 * - The underlying `fastnum::RunningStats<T>` must provide:
 *   - `push(T)`
 *   - `count() -> std::size_t`
 *   - `mean() -> T`
 *   - `variance_population() -> T`
 *   - `merge(const RunningStats&)`
 *   - `reset()`
 *
 * ## Readiness / policy
 * Standardization is only meaningful once variance is defined and non-trivial.
 * This class considers itself "ready" when:
 * - at least 2 samples have been observed,
 * - population variance is not NaN,
 * - population variance is larger than `eps_^2`.
 *
 * If not ready:
 * - `transform(x)` returns `NaN`
 * - `transform_inplace(...)` fills outputs with `NaN`
 *
 * This explicit NaN policy makes downstream problems easy to detect
 * (instead of silently returning zeros).
 *
 * ## Numerical notes
 * - Uses population variance (\f$\sigma^2 = E[(x-\mu)^2]\f$).
 * - Scaling uses `1 / sqrt(variance_population())`.
 *
 * ## Complexity
 * - `observe(x)`: O(1)
 * - `observe(batch)`: O(n)
 * - `transform(x)`: O(1)
 * - `transform_inplace(batch)`: O(n)
 * - `merge(other)`: depends on `RunningStats::merge` (typically O(1))
 *
 * @tparam T Floating-point type for accumulation and output.
 */
template <typename T = double>
class OnlineStandardScaler {
    static_assert(std::is_floating_point_v<T>,
                  "OnlineStandardScaler requires floating point T");

public:
    /**
     * @brief Observe a single sample and update running statistics.
     *
     * This updates the internal running mean and variance estimates.
     *
     * @param x Sample value.
     */
    constexpr void observe(T x) noexcept {
        stats_.push(x);
    }

    /**
     * @brief Observe a batch of samples given by pointer + length.
     *
     * Safe no-op if `xs == nullptr` or `n == 0`.
     *
     * @param xs Pointer to first sample.
     * @param n  Number of samples.
     */
    constexpr void observe(const T* xs, std::size_t n) noexcept {
        if (!xs || n == 0) return;
        for (std::size_t i = 0; i < n; ++i) {
            stats_.push(xs[i]);
        }
    }

    /**
     * @brief Observe a batch from any container with `.data()` and `.size()`.
     *
     * This overload supports types like `std::vector<T>`, `std::array<T, N>`,
     * and other contiguous containers exposing `data()` and `size()`.
     *
     * The container element type should be convertible to `T`.
     *
     * @tparam Container Any type supporting `c.data()` and `c.size()`.
     * @param c Container of samples.
     */
    template <class Container>
    constexpr auto observe(const Container& c) noexcept
        -> decltype(c.data(), c.size(), void()) {
        observe(c.data(), static_cast<std::size_t>(c.size()));
    }

    /**
     * @brief Whether the scaler is ready to produce meaningful standardized values.
     *
     * @return true if at least 2 samples have been seen and variance is
     *         non-NaN and greater than `eps_^2`.
     */
    [[nodiscard]] constexpr bool ready() const noexcept {
        if (stats_.count() < 2) return false;
        T pop_var = stats_.variance_population();
        if (std::isnan(pop_var)) return false;
        if (pop_var <= eps_ * eps_) return false;
        return true;
    }

    /**
     * @brief Number of samples observed so far.
     *
     * @return Running sample count.
     */
    [[nodiscard]] constexpr std::size_t count() const noexcept {
        return stats_.count();
    }

    /**
     * @brief Current running mean estimate.
     *
     * This value is updated online as samples are observed.
     *
     * @return Mean of observed samples (as tracked by RunningStats).
     */
    [[nodiscard]] constexpr T mean() const noexcept {
        return stats_.mean();
    }

    /**
     * @brief Transform a single value to its z-score using current running stats.
     *
     * If the scaler is not `ready()`, this returns `NaN`.
     *
     * @param x Value to standardize.
     * @return Standardized value `(x - mean) / stddev`, or `NaN` if not ready.
     */
    [[nodiscard]] T transform(T x) const noexcept {
        if (!ready()) return std::numeric_limits<T>::quiet_NaN();
        T inv_std = T{1} / std::sqrt(stats_.variance_population());
        return (x - stats_.mean()) * inv_std;
    }

    /**
     * @brief Standardize a batch of values in-place using pointer + length.
     *
     * Safe no-op if `xs == nullptr` or `n == 0`.
     *
     * If the scaler is not `ready()`, the batch is filled with `NaN`.
     *
     * @param xs Pointer to first element (modified in-place).
     * @param n  Number of elements.
     */
    void transform_inplace(T* xs, std::size_t n) const noexcept {
        if (!xs || n == 0) return;

        if (!ready()) {
            for (std::size_t i = 0; i < n; ++i) {
                xs[i] = std::numeric_limits<T>::quiet_NaN();
            }
            return;
        }

        const T mu = stats_.mean();
        const T inv_std = T{1} / std::sqrt(stats_.variance_population());
        for (std::size_t i = 0; i < n; ++i) {
            xs[i] = (xs[i] - mu) * inv_std;
        }
    }

    /**
     * @brief Standardize a contiguous container in-place via `.data()`/`.size()`.
     *
     * Supports containers like `std::vector<T>` and `std::array<T, N>`.
     *
     * If the scaler is not `ready()`, elements are filled with `NaN`.
     *
     * @tparam Container Any type supporting `c.data()` and `c.size()`.
     * @param c Container of values to standardize (modified in-place).
     */
    template <class Container>
    auto transform_inplace(Container& c) const noexcept
        -> decltype(c.data(), c.size(), void()) {
        transform_inplace(c.data(), static_cast<std::size_t>(c.size()));
    }

    /**
     * @brief Merge another fitted scaler into this one.
     *
     * This enables parallel fitting:
     * - Fit one scaler per shard/thread/partition
     * - Merge them to obtain global running statistics
     *
     * Correctness depends on `RunningStats::merge` combining counts, means,
     * and variances appropriately.
     *
     * @param other Another scaler to merge into this one.
     */
    constexpr void merge(const OnlineStandardScaler& other) noexcept {
        stats_.merge(other.stats_);
    }

    /**
     * @brief Reset state back to "unfitted".
     *
     * After reset:
     * - `count() == 0`
     * - `ready() == false`
     * - `mean()` and variance are whatever `RunningStats::reset()` defines
     *
     * Note: `transform`/`transform_inplace` will return/fill NaNs until
     * enough new samples are observed.
     */
    constexpr void reset() noexcept {
        stats_.reset();
    }

private:
    /// Running statistics accumulator (mean, variance, count).
    fastnum::RunningStats<T> stats_{};

    /**
     * @brief Variance floor control.
     *
     * `ready()` requires variance_population() > eps_^2.
     * This avoids division by ~0 in cases of constant/near-constant streams.
     */
    T eps_{static_cast<T>(1e-12)};
};

} // namespace fastnum
