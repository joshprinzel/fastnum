#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <fastnum/online_standard_scaler.hpp>

#include <random>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstddef>

TEST_CASE("OnlineStandardScaler readiness") {
    fastnum::OnlineStandardScaler<double> scaler;

    REQUIRE_FALSE(scaler.ready());

    scaler.observe(1.0);
    REQUIRE_FALSE(scaler.ready()); // count < 2

    scaler.observe(1.0);
    REQUIRE_FALSE(scaler.ready()); // variance is 0 -> not ready

    scaler.observe(2.0);
    REQUIRE(scaler.ready());       // variance > 0 -> ready
}

TEST_CASE("OnlineStandardScaler random interval readiness") {
    fastnum::OnlineStandardScaler<double> scaler;

    std::mt19937 rng(12345);
    std::normal_distribution<double> dist(0.0, 4.0);

    REQUIRE_FALSE(scaler.ready());

    bool became_ready = false;
    int ready_at = -1;

    for (int i = 0; i < 100; ++i) {
        scaler.observe(dist(rng));

        if (!became_ready && scaler.ready()) {
            became_ready = true;
            ready_at = i; // first index where ready became true
        }

        // If we ever became ready, we should stay ready for this distribution.
        if (became_ready) {
            REQUIRE(scaler.ready());
        }
    }

    REQUIRE(became_ready);
    REQUIRE(ready_at >= 1);            // should require at least 2 samples
    REQUIRE(scaler.count() == 100);

    // Sanity: mean should be a number after observing samples
    REQUIRE_FALSE(std::isnan(scaler.mean()));
}

TEST_CASE("OnlineStandardScaler stream vs. batch equivalence") {
    constexpr std::size_t N = 1000;

    std::mt19937 rng(12345);
    std::normal_distribution<double> dist(0.0, 4.0);

    std::vector<double> data(N);
    std::generate(data.begin(), data.end(), [&]() { return dist(rng); });

    // Batch: pointer + length (matches new header)
    fastnum::OnlineStandardScaler<double> batch_scaler;
    batch_scaler.observe(data.data(), data.size());

    // Stream: one-by-one
    fastnum::OnlineStandardScaler<double> stream_scaler;
    for (double x : data) {
        stream_scaler.observe(x);
    }

    REQUIRE(batch_scaler.count() == N);
    REQUIRE(stream_scaler.count() == N);

    REQUIRE(batch_scaler.ready());
    REQUIRE(stream_scaler.ready());

    REQUIRE(batch_scaler.mean() ==
            Catch::Approx(stream_scaler.mean()).margin(1e-12));

    // Compare transforms for a few points
    for (std::size_t i = 0; i < 10; ++i) {
        const double x = data[i];
        REQUIRE(batch_scaler.transform(x) ==
                Catch::Approx(stream_scaler.transform(x)).margin(1e-12));
    }
}

TEST_CASE("OnlineStandardScaler transform_inplace matches transform") {
    constexpr std::size_t N = 256;

    std::mt19937 rng(12345);
    std::normal_distribution<double> dist(0.0, 4.0);

    std::vector<double> data(N);
    std::generate(data.begin(), data.end(), [&]() { return dist(rng); });

    fastnum::OnlineStandardScaler<double> scaler;
    scaler.observe(data.data(), data.size());
    REQUIRE(scaler.ready());

    // copy -> inplace transform
    std::vector<double> inplace = data;
    scaler.transform_inplace(inplace.data(), inplace.size());

    // elementwise compare with scalar transform
    for (std::size_t i = 0; i < N; ++i) {
        REQUIRE(inplace[i] == Catch::Approx(scaler.transform(data[i])).margin(1e-12));
    }
}

TEST_CASE("OnlineStandardScaler not-ready policy returns NaN") {
    fastnum::OnlineStandardScaler<double> scaler;

    // Not ready: no samples
    REQUIRE_FALSE(scaler.ready());
    REQUIRE(std::isnan(scaler.transform(1.0)));

    // Not ready: constant samples -> variance 0
    scaler.observe(5.0);
    scaler.observe(5.0);
    REQUIRE_FALSE(scaler.ready());
    REQUIRE(std::isnan(scaler.transform(5.0)));

    std::vector<double> xs{1.0, 2.0, 3.0};
    scaler.transform_inplace(xs.data(), xs.size());
    for (double v : xs) {
        REQUIRE(std::isnan(v));
    }
}
