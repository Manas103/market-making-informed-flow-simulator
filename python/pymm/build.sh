#!/bin/bash
set -e
NB_DIR=$(python3 -c "import nanobind; print(nanobind.cmake_dir())")
mkdir -p ~/build/pymm
cd ~/build/pymm
cmake -DCMAKE_BUILD_TYPE=Release -Dnanobind_DIR="$NB_DIR" \
  "/mnt/c/Users/Manas/Downloads/Auto Job Applier/Total Job Application process flow/projects/market-making-informed-flow-simulator/python/pymm"
make -j"$(( $(nproc) / 2 ))"
