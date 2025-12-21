#pragma once
#include <cstddef>
#include <limits>
#include <span>
#include <type_traits>
#include <cmath>
#include <fastnum/running_stats.hpp>

namespace fastnum{
    template <typename T = double>
    
    class OnlineStandardScaler{
        static_assert(std::is_floating_point_v<T>, "OnlineStandardScaler requires floating point T");
        public:
        //Algorithm we are trying to implement: z = (x - mu)/stddev
        
        //Update the running estimates with one sample
        constexpr void observe(T x) noexcept{
            stats_.push(x);

        }

        //Update with a batch of samples
        constexpr void observe(std::span<const T> xs) noexcept{
            for(auto x : xs){
                stats_.push(x);
            }

        }

        //True if variance is defined and scaling is meaningful
        [[nodiscard]] constexpr bool ready() const noexcept{
            if(stats_.count() < 2) return false;
            T pop_var = stats_.variance_population();
            if(std::isnan(pop_var)) return false;
            if(pop_var <= eps_*eps_) return false;
            return true;
        }
        
        [[nodiscard]] constexpr std::size_t count() const noexcept{
            return stats_.count();

        }

        [[nodiscard]] constexpr T mean() const noexcept{
            return stats_.mean();

        }

        //Returns z-score; if not ready, returns NaNs
       [[nodiscard]] T transform (T x) const noexcept{
            if(!ready()) return std::numeric_limits<T>::quiet_NaN();
            T inverse_std = T{1} / std::sqrt(stats_.variance_population());
            return (x - stats_.mean()) * inverse_std;
        }

        //In-place standardization; no-op or fill with NaNs if not ready (your policy)
        void transform_inplace(std::span<T> xs) const noexcept{
            if(!ready()){
                for(auto& x: xs){
                    x = std::numeric_limits<T>::quiet_NaN();
                }
                return;
            }
            T mean = stats_.mean();
            T inverse_std = T{1} / std::sqrt(stats_.variance_population());
            //Loop through and fill in each element of xs.
            for(auto& x : xs){
                x = (x-mean) * inverse_std;
            }
            return;

        }

        //Merge two scalers (parallel / shared fitting)
        constexpr void merge(const OnlineStandardScaler& other) noexcept{
            stats_.merge(other.stats_);

        }

        constexpr void reset() noexcept{
            stats_.reset();
        }

        private:
        fastnum::RunningStats<T> stats_{};
        T eps_{static_cast<T>(1e-12)};

    };
}