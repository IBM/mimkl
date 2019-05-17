#!/bin/bash
sources_directory=$1
build_directory=$2
python_executable=$3
processes=$4

cd $build_directory
cmake -DPYTHON_EXECUTABLE=$python_executable -D CMAKE_BUILD_TYPE=Release $sources_directory
cmake --build . --config Release --target all -- -j $processes
