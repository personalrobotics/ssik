# ssik native C++ artifacts + conformance (#482 Phase 0a).
cmake_minimum_required(VERSION 3.16)
project(ssik_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Eigen (header-only). Prefer the CMake package; fall back to common include
# paths so a bare `cmake ..` works without Eigen3Config on the path.
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(TARGET Eigen3::Eigen)
  set(SSIK_EIGEN Eigen3::Eigen)
else()
  find_path(EIGEN3_INCLUDE_DIR Eigen/Dense
    PATHS /opt/homebrew/include/eigen3 /usr/local/include/eigen3 /usr/include/eigen3)
  if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Eigen not found; install eigen (brew install eigen / apt install libeigen3-dev)")
  endif()
  add_library(ssik_eigen INTERFACE)
  target_include_directories(ssik_eigen INTERFACE ${EIGEN3_INCLUDE_DIR})
  set(SSIK_EIGEN ssik_eigen)
endif()

# Shared native headers (Band C primitives) + the generated per-arm artifacts.
add_library(ssik_cpp INTERFACE)
target_include_directories(ssik_cpp INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/gen)
target_link_libraries(ssik_cpp INTERFACE ${SSIK_EIGEN})

enable_testing()

# The conformance tests include the parity fixtures (cpp/gen/*_parity.hpp), which
# are git-ignored and regenerated from the current Python oracle (#82/#495):
#   python scripts/cpp_emit.py --all
# Run that before configuring/building the tests from a fresh checkout. Shipped
# consumers use only the constants header + include/ primitives, not these tests.

add_executable(test_ur5_fk tests/test_ur5_fk.cpp)
target_link_libraries(test_ur5_fk PRIVATE ssik_cpp)
add_test(NAME ur5_fk COMMAND test_ur5_fk)

# Solve-level conformance is validated by the reused Python suite (the C++
# backend runs tests/test_three_parallel.py through the pybind binding, #499),
# not a hand-written C++ set-agreement test. These two goldens validate the FK
# and Newton primitives Python-free.
add_executable(test_ur5_newton tests/test_ur5_newton.cpp)
target_link_libraries(test_ur5_newton PRIVATE ssik_cpp)
add_test(NAME ur5_newton COMMAND test_ur5_newton)

# THE GATE (data-driven): every self-contained artifact validated against its
# Python oracle golden, zero pybind/Python. The arm list is generated into
# cpp/gen/artifact_gate.hpp by cpp_emit.py, so onboarding an arm needs no edit
# here. Covers three_parallel (ur5), spherical_two_parallel (irb6700), and
# seven_r.srs canonical (iiwa14) + general Davenport (iiwa7/r1pro/openarm) paths.
add_executable(test_artifacts tests/test_artifacts.cpp)
target_link_libraries(test_artifacts PRIVATE ssik_cpp)
# The generated gate (cpp/gen/artifact_gate.hpp) includes artifact_conformance.hpp
# from cpp/tests.
target_include_directories(test_artifacts PRIVATE tests)
add_test(NAME artifacts COMMAND test_artifacts)
