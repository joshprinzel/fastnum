#pragma once

#include <cstddef>
#include <limits>
#include <type_traits>
#include <cmath>
#include <cassert>

namespace fastnum {

template <typename T = double>
class OnlineCovariance {
    static_assert(std::is_floating_point_v<T>, "OnlineCovariance requires floating point T");

public:
    // --- Observe -------------------------------------------------------------

    constexpr void observe(T x, T y) noexcept {
        ++n_;

        // Save deltas against the *old* means
        const T dx = x - mean_x_;
        const T dy = y - mean_y_;

        // Update means
        const T inv_n = T{1} / static_cast<T>(n_);
        mean_x_ += dx * inv_n;
        mean_y_ += dy * inv_n;

        // Deltas against the *new* means
        const T dx2 = x - mean_x_;
        const T dy2 = y - mean_y_;

        // Update second central moments (Welford)
        m2_x_ += dx * dx2;
        m2_y_ += dy * dy2;

        // Cross moment (covariance numerator)
        c_ += dx * dy2;
    }

    constexpr void observe(const T* xs, const T* ys, std::size_t n) noexcept {
        if (!xs || !ys || n == 0) return;
        for (std::size_t i = 0; i < n; ++i) observe(xs[i], ys[i]);
    }

    template <class CX, class CY>
    constexpr auto observe(const CX& xs, const CY& ys) noexcept
        -> decltype(xs.data(), xs.size(), ys.data(), ys.size(), void()) {
        assert(static_cast<std::size_t>(xs.size()) == static_cast<std::size_t>(ys.size()));
        observe(xs.data(), ys.data(), static_cast<std::size_t>(xs.size()));
    }

    // --- Basic accessors -----------------------------------------------------

    [[nodiscard]] constexpr std::size_t count() const noexcept { return n_; }
    [[nodiscard]] constexpr T mean_x() const noexcept { return mean_x_; }
    [[nodiscard]] constexpr T mean_y() const noexcept { return mean_y_; }

    // --- Variances / Covariance ---------------------------------------------

    [[nodiscard]] constexpr T variance_x_population() const noexcept {
        if (n_ < 1) return std::numeric_limits<T>::quiet_NaN();
        return m2_x_ / static_cast<T>(n_);
    }

    [[nodiscard]] constexpr T variance_y_population() const noexcept {
        if (n_ < 1) return std::numeric_limits<T>::quiet_NaN();
        return m2_y_ / static_cast<T>(n_);
    }

    [[nodiscard]] constexpr T variance_x_sample() const noexcept {
        if (n_ < 2) return std::numeric_limits<T>::quiet_NaN();
        return m2_x_ / static_cast<T>(n_ - 1);
    }

    [[nodiscard]] constexpr T variance_y_sample() const noexcept {
        if (n_ < 2) return std::numeric_limits<T>::quiet_NaN();
        return m2_y_ / static_cast<T>(n_ - 1);
    }

    [[nodiscard]] constexpr T covariance_population() const noexcept {
        if (n_ < 1) return std::numeric_limits<T>::quiet_NaN();
        return c_ / static_cast<T>(n_);
    }

    [[nodiscard]] constexpr T covariance_sample() const noexcept {
        if (n_ < 2) return std::numeric_limits<T>::quiet_NaN();
        return c_ / static_cast<T>(n_ - 1);
    }

    // --- Readiness / policy --------------------------------------------------

    [[nodiscard]] constexpr bool ready() const noexcept {
        if (n_ < 2) return false;
        const T vx = variance_x_population();
        const T vy = variance_y_population();
        if (std::isnan(vx) || std::isnan(vy)) return false;
        if (vx <= eps_ * eps_ || vy <= eps_ * eps_) return false;
        return true;
    }

    // --- Reset ---------------------------------------------------------------

    constexpr void reset() noexcept {
        n_ = 0;
        mean_x_ = T{0};
        mean_y_ = T{0};
        m2_x_ = T{0};
        m2_y_ = T{0};
        c_ = T{0};
    }

        // --- Correlation ---------------------------------------------------------

    [[nodiscard]] T correlation() const noexcept {
        if (!ready()) return std::numeric_limits<T>::quiet_NaN();

        const T vx = variance_x_population();
        const T vy = variance_y_population();
        const T denom = std::sqrt(vx * vy);

        if (std::isnan(denom) || denom <= eps_) {
            return std::numeric_limits<T>::quiet_NaN();
        }

        // Population vs sample cancels in correlation, so this is fine.
        return covariance_population() / denom;
    }


    constexpr void merge(const OnlineCovariance& other) noexcept{
        if(other.n_ == 0) return;
        if(n_ == 0){
            *this = other;
            return;
        }

        const std::size_t n_a = n_;
        const std::size_t n_b = other.n_;
        const std::size_t n = n_a + n_b;

        const T mean_x_a = mean_x_;
        const T mean_y_a = mean_y_;

        const T mean_x_b = other.mean_x_;
        const T mean_y_b = other.mean_y_;

        const T dx = mean_x_b - mean_x_a;
        const T dy = mean_y_b - mean_y_a;

        const T n_a_t = static_cast<T>(n_a);
        const T n_b_t = static_cast<T>(n_b);
        const T n_t = static_cast<T>(n);

        //Update means
        mean_x_ = mean_x_a + dx * (n_b_t / n_t);
        mean_y_ = mean_y_a + dy * (n_b_t / n_t);


        //Combine second central moments (Welford Merge)
        m2_x_ = m2_x_ + other.m2_x_ + dx * dx * (n_a_t * n_b_t / n_t);
        m2_y_ = m2_y_ + other.m2_y_ + dy * dy * (n_a_t * n_b_t / n_t);

        //Combine cross_deviation sum
        c_ = c_ + other.c_ + dx * dy * (n_a_t * n_b_t / n_t);

        n_ = n;
    }

private:
    std::size_t n_{0};
    T mean_x_{0};
    T mean_y_{0};
    T m2_x_{0};
    T m2_y_{0};
    T c_{0};
    T eps_{static_cast<T>(1e-12)};
};

} // namespace fastnum
