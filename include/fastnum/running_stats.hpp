#pragma once

#include <cstddef>
#include <limits>
#include <cmath>
#include <type_traits>

namespace fastnum {

    template <typename T = double>
    class RunningStats {
    public:
    constexpr void push(T x) noexcept {
        ++n_;
        const T delta = x - mean_;
        mean_ += delta / static_cast<T>(n_);
        const T delta2 = x - mean_;
        m2_ += delta * delta2;
    }

    // Merge another accumulator into this one (parallel-friendly).
    constexpr void merge(const RunningStats& other) noexcept {
        if (other.n_ == 0) return;
        if (n_ == 0) { *this = other; return; }

        const T n_a = static_cast<T>(n_);
        const T n_b = static_cast<T>(other.n_);
        const T n_total = n_a + n_b;

        const T delta = other.mean_ - mean_;
        mean_ = (n_a * mean_ + n_b * other.mean_) / n_total;
        m2_ += other.m2_ + (delta * delta) * (n_a * n_b / n_total);
        n_ += other.n_;
    }

    [[nodiscard]] constexpr std::size_t count() const noexcept { return n_; }
    [[nodiscard]] constexpr T mean() const noexcept { return mean_; }

    [[nodiscard]] constexpr T variance_population() const noexcept {
        if (n_ < 1) return std::numeric_limits<T>::quiet_NaN();
        return m2_ / static_cast<T>(n_);
    }

    [[nodiscard]] constexpr T variance_sample() const noexcept {
        if (n_ < 2) return std::numeric_limits<T>::quiet_NaN();
        return m2_ / static_cast<T>(n_ - 1);
    }

    [[nodiscard]] T stddev_population() const noexcept {
        const T v = variance_population();
        return std::sqrt(v);
    }

    [[nodiscard]] T stddev_sample() const noexcept {
        const T v = variance_sample();
        return std::sqrt(v);
    }

    constexpr void reset() noexcept {
        n_ = 0;
        mean_ = T{0};
        m2_ = T{0};
    }

    private:
    std::size_t n_{0};
    T mean_{0};
    T m2_{0};
    };

}  // namespace fastnum
