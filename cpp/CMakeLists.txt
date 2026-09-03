# ssik native C++ artifacts + conformance (#482 Phase 0a).
cmake_minimum_required(VERSION 3.16)
project(ssik_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Eigen (header-only). Prefer the CMake package (needed to export a clean
# ssik::ssik_cpp for downstream find_package); fall back to common include paths
# so a bare in-tree `cmake ..` still builds without Eigen3Config on the path. No
# version pin: ssik uses only stable Eigen basics, and pinning a MAJOR (e.g. 3.3)
# makes SameMajorVersion reject a newer Eigen (5.x on current Homebrew).
# std::thread (parallel.hpp, #546) needs the platform thread library linked.
find_package(Threads REQUIRED)
find_package(Eigen3 QUIET NO_MODULE)
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
# BUILD_INTERFACE points at the source tree (for the in-tree tests/examples);
# INSTALL_INTERFACE points at the installed include dir (for external consumers
# via find_package(ssik_cpp) -- see the install rules below and cpp/README.md).
include(GNUInstallDirs)  # defines CMAKE_INSTALL_INCLUDEDIR used just below
add_library(ssik_cpp INTERFACE)
add_library(ssik::ssik_cpp ALIAS ssik_cpp)
target_include_directories(ssik_cpp INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gen>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
# parallel.hpp uses std::thread (jointlock sweep / rescue / batch, #546), so every
# consumer of the headers links the platform thread library.
target_link_libraries(ssik_cpp INTERFACE ${SSIK_EIGEN} Threads::Threads)
target_compile_features(ssik_cpp INTERFACE cxx_std_20)

# Install + export the header-only package so a downstream C++/MoveIt project can
#   find_package(ssik_cpp)  ->  target_link_libraries(app ssik::ssik_cpp)
# and #include "<arm>_ik.hpp". Requires the Eigen3 CMake package (the exported
# config find_dependency()s it); the bare-include-path fallback above is in-tree
# only. Toggle with -DSSIK_CPP_INSTALL=OFF.
option(SSIK_CPP_INSTALL "Install headers + export the ssik::ssik_cpp CMake package" ON)
if(SSIK_CPP_INSTALL AND TARGET Eigen3::Eigen)
  include(CMakePackageConfigHelpers)
  # Header-only primitives + the committed per-arm artifacts (constants headers).
  # The *_parity / artifact_gate fixtures are test-only (git-ignored) -- not shipped.
  install(DIRECTORY include/ssik_cpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(DIRECTORY gen/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
    PATTERN "*_parity.hpp" EXCLUDE
    PATTERN "artifact_gate.hpp" EXCLUDE)
  install(TARGETS ssik_cpp EXPORT ssik_cppTargets)
  install(EXPORT ssik_cppTargets NAMESPACE ssik:: FILE ssik_cppTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ssik_cpp)
  configure_package_config_file(cmake/ssik_cppConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ssik_cppConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ssik_cpp)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ssik_cppConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ssik_cpp)
endif()

# In-tree usage example (built against the source-tree headers). External
# consumers build the same source against the INSTALLED package -- see
# examples/consumer/. Toggle with -DSSIK_CPP_EXAMPLES=OFF.
option(SSIK_CPP_EXAMPLES "Build the in-tree usage example" ON)
if(SSIK_CPP_EXAMPLES)
  add_executable(solve_arm examples/solve_arm.cpp)
  target_link_libraries(solve_arm PRIVATE ssik::ssik_cpp)
endif()

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

# parallel_for primitive (#546): coverage, no-race, worker-count independence.
add_executable(test_parallel tests/test_parallel.cpp)
target_link_libraries(test_parallel PRIVATE ssik_cpp Threads::Threads)
add_test(NAME parallel COMMAND test_parallel)

# solve_batch equivalence (#546): parallel batch == serial per-pose solve.
add_executable(test_batch tests/test_batch.cpp)
target_link_libraries(test_batch PRIVATE ssik_cpp)
target_include_directories(test_batch PRIVATE gen)
add_test(NAME batch COMMAND test_batch)

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
