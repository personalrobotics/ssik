# ssik::ssik_cpp for the copy of the headers inside the ssik wheel. hatch_build.py
# ships this file as ssik/cpp/cmake/ssik_cppTargets.cmake, next to the config
# rendered from ssik_cppConfig.cmake.in, and the headers in ssik/cpp/include.
# It defines the target `cmake --install` exports (see CMakeLists.txt) with the
# same interface: header-only, C++20, Eigen3::Eigen and Threads::Threads, which
# the config has already found. The include directory is found relative to this
# file, so the package works wherever the wheel is installed.
if(TARGET ssik::ssik_cpp)
  return()
endif()

get_filename_component(_ssik_cpp_include "${CMAKE_CURRENT_LIST_DIR}/../include" ABSOLUTE)
if(NOT EXISTS "${_ssik_cpp_include}/ssik_cpp/fk.hpp")
  message(FATAL_ERROR "ssik_cpp: headers missing from ${_ssik_cpp_include}; reinstall ssik")
endif()

add_library(ssik::ssik_cpp INTERFACE IMPORTED)
set_target_properties(ssik::ssik_cpp PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_ssik_cpp_include}"
  INTERFACE_LINK_LIBRARIES "Eigen3::Eigen;Threads::Threads"
  INTERFACE_COMPILE_FEATURES "cxx_std_20")
unset(_ssik_cpp_include)
