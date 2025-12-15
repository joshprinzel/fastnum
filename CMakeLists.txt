cmake_minimum_required(VERSION 3.20)
project(fastnum VERSION 0.1.0 LANGUAGES CXX)

# ---------- Options ----------
option(FASTNUM_BUILD_TESTS "Build fastnum tests" ON)
option(FASTNUM_BUILD_EXAMPLES "Build fastnum examples" ON)

# ---------- Library ----------
add_library(fastnum INTERFACE)
add_library(fastnum::fastnum ALIAS fastnum)

target_compile_features(fastnum INTERFACE cxx_std_20)

target_include_directories(fastnum INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Reasonable warnings for consumers (only when building this project directly)
if (PROJECT_IS_TOP_LEVEL)
  if (MSVC)
    target_compile_options(fastnum INTERFACE /W4)
  else()
    target_compile_options(fastnum INTERFACE -Wall -Wextra -Wpedantic)
  endif()
endif()

# ---------- Tests ----------
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

  add_executable(fastnum_tests
    tests/test_running_stats.cpp
  )
  target_link_libraries(fastnum_tests PRIVATE fastnum::fastnum Catch2::Catch2WithMain)

  add_test(NAME fastnum_tests COMMAND fastnum_tests)
endif()

# ---------- Examples ----------
if (FASTNUM_BUILD_EXAMPLES)
  add_executable(demo_running_stats examples/demo_running_stats.cpp)
  target_link_libraries(demo_running_stats PRIVATE fastnum::fastnum)
endif()
