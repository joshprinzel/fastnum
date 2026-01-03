cmake_minimum_required(VERSION 3.16)
project(fastnum VERSION 0.1.0 LANGUAGES CXX)

# When fastnum built by itself -> ON
# When fastnum is included in another project -> OFF
option(FASTNUM_BUILD_TESTS "Build fastnum tests" ${PROJECT_IS_TOP_LEVEL})
option(FASTNUM_BUILD_EXAMPLES "Build fastnum examples" ${PROJECT_IS_TOP_LEVEL})
option(FASTNUM_BUILD_BENCHMARKS "Build fastnum benchmarks" ${PROJECT_IS_TOP_LEVEL})

# Default to Release for single-config generators
if (PROJECT_IS_TOP_LEVEL AND NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(fastnum INTERFACE)
add_library(fastnum::fastnum ALIAS fastnum)
target_compile_features(fastnum INTERFACE cxx_std_20)

target_include_directories(fastnum INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Build-only warnings (do not leak to consumers)
add_library(fastnum_warnings INTERFACE)
if (MSVC)
  target_compile_options(fastnum_warnings INTERFACE /W4)
else()
  target_compile_options(fastnum_warnings INTERFACE -Wall -Wextra -Wpedantic)
endif()

if (FASTNUM_BUILD_TESTS)
  include(CTest)
  enable_testing()

  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.4
  )
  FetchContent_MakeAvailable(Catch2)

  file(GLOB TEST_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp
  )

  add_executable(fastnum_tests ${TEST_SOURCES})
  target_link_libraries(fastnum_tests PRIVATE fastnum::fastnum fastnum_warnings Catch2::Catch2WithMain)

  list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
  include(Catch)
  catch_discover_tests(fastnum_tests)
endif()

if (FASTNUM_BUILD_BENCHMARKS)
  include(FetchContent)
  FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
  )
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)

  file(GLOB BENCH_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/bench_*.cpp
  )

  add_executable(fastnum_bench ${BENCH_SOURCES})
  target_link_libraries(fastnum_bench PRIVATE fastnum::fastnum fastnum_warnings benchmark::benchmark)
  target_compile_definitions(fastnum_bench PRIVATE NDEBUG)

  if (MSVC)
    target_compile_options(fastnum_bench PRIVATE /O2)
  else()
    target_compile_options(fastnum_bench PRIVATE -O3 -fno-omit-frame-pointer)
  endif()
endif()

if (FASTNUM_BUILD_EXAMPLES)
  add_executable(demo_running_stats examples/demo_running_stats.cpp)
  target_link_libraries(demo_running_stats PRIVATE fastnum::fastnum fastnum_warnings)
endif()
