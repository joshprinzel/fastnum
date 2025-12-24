#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <fastnum/online_covariance.hpp>

#include <random>
#include <vector>
#include <numeric>
#include <cmath>
#include <limits>

// --- Naive reference implementations (slow but correct) ---

static double naive_mean(const std::vector<double>& xs) {
    return std::accumulate(xs.begin(), xs.end(), 0.0) /
           static_cast<double>(xs.size());
}

static double naive_cov_population(const std::vector<double>& xs,
                                   const std::vector<double>& ys) {
    if (xs.empty()) return std::numeric_limits<double>::quiet_NaN();
    const double mx = naive_mean(xs);
    const double my = naive_mean(ys);

    double sum = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        sum += (xs[i] - mx) * (ys[i] - my);
    }
    return sum / static_cast<double>(xs.size());
}

static double naive_cov_sample(const std::vector<double>& xs,
                               const std::vector<double>& ys) {
    if (xs.size() < 2) return std::numeric_limits<double>::quiet_NaN();
    const double mx = naive_mean(xs);
    const double my = naive_mean(ys);

    double sum = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        sum += (xs[i] - mx) * (ys[i] - my);
    }
    return sum / static_cast<double>(xs.size() - 1);
}

static double naive_var_population(const std::vector<double>& xs) {
    if (xs.empty()) return std::numeric_limits<double>::quiet_NaN();
    const double mx = naive_mean(xs);
    double sum = 0.0;
    for (double x : xs) {
        const double d = x - mx;
        sum += d * d;
    }
    return sum / static_cast<double>(xs.size());
}

static double naive_corr(const std::vector<double>& xs,
                         const std::vector<double>& ys) {
    if (xs.size() < 2) return std::numeric_limits<double>::quiet_NaN();
    const double cov = naive_cov_population(xs, ys);
    const double vx = naive_var_population(xs);
    const double vy = naive_var_population(ys);
    const double denom = std::sqrt(vx * vy);
    if (std::isnan(denom) || denom == 0.0) return std::numeric_limits<double>::quiet_NaN();
    return cov / denom;
}

TEST_CASE("OnlineCovariance readiness and NaN policy", "[covariance]") {
    fastnum::OnlineCovariance<double> cov;

    REQUIRE(cov.count() == 0);
    REQUIRE_FALSE(cov.ready());
    REQUIRE(std::isnan(cov.covariance_population()));
    REQUIRE(std::isnan(cov.covariance_sample()));
    REQUIRE(std::isnan(cov.correlation()));

    cov.observe(1.0, 2.0);
    REQUIRE(cov.count() == 1);
    REQUIRE_FALSE(cov.ready());
    REQUIRE(std::isnan(cov.covariance_sample()));
    REQUIRE(std::isnan(cov.correlation()));

    cov.observe(2.0, 3.0);
    REQUIRE(cov.count() == 2);
    REQUIRE(cov.ready());
    REQUIRE_FALSE(std::isnan(cov.correlation()));
}

TEST_CASE("OnlineCovariance matches naive reference on random data", "[covariance]") {
    std::mt19937 rng(12345);
    std::normal_distribution<double> dist(0.0, 3.0);

    for (int trial = 0; trial < 200; ++trial) {
        const int n = 2 + (trial % 200);

        std::vector<double> xs;
        std::vector<double> ys;
        xs.reserve(n);
        ys.reserve(n);

        for (int i = 0; i < n; ++i) {
            const double x = dist(rng);
            const double y = 0.8 * x + 0.2 * dist(rng); // correlated-ish
            xs.push_back(x);
            ys.push_back(y);
        }

        fastnum::OnlineCovariance<double> cov;
        cov.observe(xs, ys);

        REQUIRE(cov.count() == xs.size());
        REQUIRE(cov.mean_x() == Catch::Approx(naive_mean(xs)).epsilon(1e-12));
        REQUIRE(cov.mean_y() == Catch::Approx(naive_mean(ys)).epsilon(1e-12));

        REQUIRE(cov.covariance_population() ==
                Catch::Approx(naive_cov_population(xs, ys)).epsilon(1e-10));

        REQUIRE(cov.covariance_sample() ==
                Catch::Approx(naive_cov_sample(xs, ys)).epsilon(1e-10));

        REQUIRE(cov.correlation() ==
                Catch::Approx(naive_corr(xs, ys)).epsilon(1e-10));
    }
}

TEST_CASE("OnlineCovariance merge equals observe-all-at-once", "[covariance][merge]") {
    std::mt19937 rng(6789);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    for (int trial = 0; trial < 200; ++trial) {
        const int n = 2 + (trial % 300);

        std::vector<double> xs(n), ys(n);
        for (int i = 0; i < n; ++i) {
            xs[i] = dist(rng);
            ys[i] = 2.0 * xs[i] + 0.5 * dist(rng);
        }

        // observe-all-at-once
        fastnum::OnlineCovariance<double> all;
        all.observe(xs, ys);

        // split + merge
        const int split = n / 2;
        fastnum::OnlineCovariance<double> a, b;
        a.observe(xs.data(), ys.data(), static_cast<std::size_t>(split));
        b.observe(xs.data() + split, ys.data() + split,
                  static_cast<std::size_t>(n - split));
        a.merge(b);

        REQUIRE(a.count() == all.count());
        REQUIRE(a.mean_x() == Catch::Approx(all.mean_x()).epsilon(1e-12));
        REQUIRE(a.mean_y() == Catch::Approx(all.mean_y()).epsilon(1e-12));
        REQUIRE(a.covariance_population() ==
                Catch::Approx(all.covariance_population()).epsilon(1e-10));
        REQUIRE(a.covariance_sample() ==
                Catch::Approx(all.covariance_sample()).epsilon(1e-10));
        REQUIRE(a.correlation() ==
                Catch::Approx(all.correlation()).epsilon(1e-10));
    }
}
