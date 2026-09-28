# Standalone downstream project: consumes ssik_cpp via find_package, exactly as a
# real C++/MoveIt user would. Built against the INSTALLED package (not the source
# tree) to prove the install/export actually works end to end. See
# examples/consumer/smoke.sh (and the "C++ consumer" CI job) for the driver.
cmake_minimum_required(VERSION 3.16)
project(ssik_consumer LANGUAGES CXX)

find_package(ssik_cpp REQUIRED)
# ssik_cpp's config find_dependency()s Eigen3. The smoke passes the pinned
# version (#606) so it proves the consumer resolved that Eigen, not another one.
set(SSIK_EIGEN_VERSION "" CACHE STRING "Require exactly this Eigen version (empty: any)")
if(SSIK_EIGEN_VERSION AND NOT Eigen3_VERSION VERSION_EQUAL SSIK_EIGEN_VERSION)
  message(FATAL_ERROR "consumer found Eigen ${Eigen3_VERSION}, expected ${SSIK_EIGEN_VERSION}")
endif()
message(STATUS "ssik_consumer: Eigen ${Eigen3_VERSION} (${Eigen3_DIR})")

add_executable(solve_arm ../solve_arm.cpp)
target_link_libraries(solve_arm PRIVATE ssik::ssik_cpp)
