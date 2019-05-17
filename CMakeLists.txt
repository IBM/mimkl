cmake_minimum_required(VERSION 3.0.2)

project(mimkl)

# project version
set(VERSION_MAJOR 0)
set(VERSION_MINOR 0)
set(VERSION_PATCH 1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_CXX_EXTENSIONS OFF)

if(DEFINED ENV{MIMKL_CXX_FLAGS})
    message(STATUS "Additional C++ compiler flags passed")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2 -ftest-coverage -fprofile-arcs -pthread -fPIC $ENV{MIMKL_CXX_FLAGS}")
else()
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2 -ftest-coverage -fprofile-arcs -pthread -fPIC")
endif()

message(STATUS "CMAKE_CXX_FLAGS is ${CMAKE_CXX_FLAGS}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
message(STATUS "CMAKE_MODULE_PATH is ${CMAKE_MODULE_PATH}")

#add third party libraries dependencies from submodules
# eigen3
include_directories(${mimkl_SOURCE_DIR}/external/eigen3)
add_subdirectory(external/eigen3)
add_library(eigen3 INTERFACE)
# dlib
include_directories(${mimkl_SOURCE_DIR}/external/dlib)
# temporarly removed to avoid compilation of useless sources
# add_subdirectory(external/dlib)
# spdlog
include_directories(${mimkl_SOURCE_DIR}/external/spdlog/include)
add_subdirectory(external/spdlog)
# pybind11
include_directories(${mimkl_SOURCE_DIR}/external/pybind11/include)
add_subdirectory(external/pybind11)


#include mimkl headers
include_directories(${mimkl_SOURCE_DIR}/include/)
add_subdirectory(src)

add_subdirectory(python)

enable_testing()
add_subdirectory(test)
# copy data for tests
file(COPY ${mimkl_SOURCE_DIR}/test/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
# ctest configuration
configure_file(${mimkl_SOURCE_DIR}/CTestCustom.cmake ${CMAKE_BINARY_DIR} @ONLY)

# add a target to generate API documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile @ONLY)
add_custom_target(mimkl_doc
${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile
WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
COMMENT "Generating API documentation with Doxygen" VERBATIM
)
endif(DOXYGEN_FOUND)
