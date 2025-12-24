#include <fastnum/running_stats.hpp>
#include <iostream>

int main() {
  fastnum::RunningStats<double> rs;
  for (int i = 1; i <= 10; ++i) rs.observe(i);

  std::cout << "n=" << rs.count()
            << " mean=" << rs.mean()
            << " var(sample)=" << rs.variance_sample()
            << "\n";
}
