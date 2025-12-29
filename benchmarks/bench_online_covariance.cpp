#include <benchmark/benchmark.h>
#include <fastnum/online_covariance.hpp>
#include <random>
#include <vector>

static void BM_observe_steady_state(benchmark::State& state){
    fastnum::OnlineCovariance<double> cov;

    //Warm up cache for ideal settings for benchmarking
    for(int i = 0; i < 1000; ++i) cov.observe(1.0,2.0);

    //What are we trying to measure? -> how long it takes to use observe() in Online Covariance Module
    for(auto _ : state){
        cov.observe(1.0, 2.0);
        //Make sure the compiler does not cheat
        benchmark::DoNotOptimize(cov.covariance_population());        
    }

    //Shows how many items processed
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_observe_steady_state);


static void BM_observe_array_streaming(benchmark::State& state){
  const std::size_t N = static_cast<std::size_t>(state.range(0));

  std::vector<double> xs(N), ys(N);
  std::mt19937_64 rng(12345);
  std::uniform_real_distribution<double> dist(1.0, 1000.0);
  for (std::size_t i = 0; i < N; ++i) { xs[i] = dist(rng); ys[i] = dist(rng); }

  fastnum::OnlineCovariance<double> cov;
  for (int i = 0; i < 1000; ++i) cov.observe(1.0, 2.0); // warm steady state

  for (auto _ : state){
    cov.observe(xs.data(), ys.data(), N);
    benchmark::DoNotOptimize(cov.covariance_population());
  }

  state.SetItemsProcessed(state.iterations() * N);

  const double bytes = double(N) * 2.0 * sizeof(double);
  state.counters["Input_GiB/s"] = benchmark::Counter(
      bytes, benchmark::Counter::kIsRate, benchmark::Counter::OneK::kIs1024);
}


BENCHMARK(BM_observe_array_streaming)->Arg(1<<10)->Arg(1<<16)->Arg(1<<20)->UseRealTime();
BENCHMARK_MAIN();