#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <fastnum/running_stats.hpp>
#include <cmath>
#include <random>
#include <numeric>
#include <limits>
#include <vector>

// --- Naive Refernce Implementations (slow but correct) ---
static double naive_mean(const std::vector<double>& xs){
    return std::accumulate(xs.begin(), xs.end(), 0.0) / static_cast<double>(xs.size());
}

static double naive_sample_var(const std::vector<double>& xs){
    if(xs.size()<2) return std::numeric_limits<double>::quiet_NaN();

    const double mu = naive_mean(xs);
    double sum = 0.0;
    for (double x : xs){
        const double d = x - mu;
        sum += d*d;
    }
    return sum / static_cast<double>(xs.size()-1);
}

TEST_CASE("RunningStats mean/variance", "[runningstats]") {
  fastnum::RunningStats<double> rs;
  rs.push(1);
  rs.push(2);
  rs.push(3);
  rs.push(4);
  rs.push(5);

  REQUIRE(rs.count() == 5);
  REQUIRE(rs.mean() == Catch::Approx(3.0));
  REQUIRE(rs.variance_sample() == Catch::Approx(2.5));
}

TEST_CASE("RunningStats matches naive on random data", "[runningstats]"){
    std::mt19937 rng(12345);
    std::normal_distribution<double> dist(0.0,3.0);

    for(int trial = 0; trial < 200; ++trial){
        const int n = 2 + (trial % 200);

        std::vector<double> xs;
        xs.reserve(n);
        for(int i = 0; i < n; ++i) xs.push_back(dist(rng));

        fastnum::RunningStats<double> rs;
        for(double x: xs) rs.push(x);

        //Are they equal in size?
        REQUIRE(rs.count() == xs.size());

        //Same mean as naive approach?
        REQUIRE(rs.mean() == Catch::Approx(naive_mean(xs)).epsilon(1e-12));

        //Same var sample as naive?
        REQUIRE(rs.variance_sample() == Catch::Approx(naive_sample_var(xs)).epsilon(1e-10));
    }
    

}

TEST_CASE("RunningStats merge equals push-all-at-once", "[runningstats][merge]") {
  std::mt19937 rng(6789);
  std::uniform_real_distribution<double> dist(-10.0, 10.0);

  for (int trial = 0; trial < 200; ++trial) {
    const int n = 2 + (trial % 300);

    std::vector<double> xs(n);
    for (double& x : xs) x = dist(rng);

    // push-all-at-once
    fastnum::RunningStats<double> all;
    for (double x : xs) all.push(x);

    // split + merge
    const int split = n / 2;
    fastnum::RunningStats<double> a, b;
    for (int i = 0; i < split; ++i) a.push(xs[i]);
    for (int i = split; i < n; ++i) b.push(xs[i]);
    a.merge(b);

    REQUIRE(a.count() == all.count());
    REQUIRE(a.mean() == Catch::Approx(all.mean()).epsilon(1e-12));
    REQUIRE(a.variance_sample() == Catch::Approx(all.variance_sample()).epsilon(1e-10));
  }
}


