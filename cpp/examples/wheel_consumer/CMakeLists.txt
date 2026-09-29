# Standalone downstream project that consumes the ssik_cpp headers shipped in
# the ssik WHEEL, found through ssik.get_cmake_dir(), with no ssik source tree.
# check.py drives it (and the "Native wheel build" CI job runs check.py):
#   cmake -S . -B build -DCMAKE_PREFIX_PATH="$(python -c 'import ssik; print(ssik.get_cmake_dir())');<eigen prefix>"
cmake_minimum_required(VERSION 3.16)
project(ssik_wheel_consumer LANGUAGES CXX)

# The ssik major version to require (check.py passes the installed one), so the
# wheel's version file is exercised too.
set(SSIK_CPP_VERSION "" CACHE STRING "ssik_cpp version to request (empty: any)")
find_package(ssik_cpp ${SSIK_CPP_VERSION} CONFIG REQUIRED)
message(STATUS "ssik_wheel_consumer: ssik_cpp ${ssik_cpp_VERSION} (${ssik_cpp_DIR})")

# Eigen is the consumer's: the smoke passes the pinned version (#606) and fails
# if the config resolved another one.
set(SSIK_EIGEN_VERSION "" CACHE STRING "Require exactly this Eigen version (empty: any)")
if(SSIK_EIGEN_VERSION AND NOT Eigen3_VERSION VERSION_EQUAL SSIK_EIGEN_VERSION)
  message(FATAL_ERROR "consumer found Eigen ${Eigen3_VERSION}, expected ${SSIK_EIGEN_VERSION}")
endif()
message(STATUS "ssik_wheel_consumer: Eigen ${Eigen3_VERSION} (${Eigen3_DIR})")

add_executable(three_parallel_solve three_parallel_solve.cpp)
target_link_libraries(three_parallel_solve PRIVATE ssik::ssik_cpp)
