# Standalone downstream project: consumes ssik_cpp via find_package, exactly as a
# real C++/MoveIt user would. Built against the INSTALLED package (not the source
# tree) to prove the install/export actually works end to end. See
# examples/consumer/smoke.sh (and the "C++ consumer" CI job) for the driver.
cmake_minimum_required(VERSION 3.16)
project(ssik_consumer LANGUAGES CXX)

find_package(ssik_cpp REQUIRED)

add_executable(solve_arm ../solve_arm.cpp)
target_link_libraries(solve_arm PRIVATE ssik::ssik_cpp)
