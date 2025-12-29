#include <benchmark/benchmark.h>
#include <fastnum/online_covariance.hpp>

static void BM_Smoke(benchmark::State& state){
    fastnum::OnlineCovariance<double> oc;
    for (auto _ : state){
        oc.observe(1.0,2.0);
        benchmark::DoNotOptimize(oc);
    }
}
BENCHMARK(BM_Smoke);
