#!/usr/bin/env bash
set -euo pipefail

# Build a standalone GDA probe binary that only depends on libfabric and CUDA.
# Usage: scripts/build_standalone_probe.sh [OUT_DIR]
# Default OUT_DIR: /workspace

OUT_DIR="${1:-/workspace}"
mkdir -p "${OUT_DIR}"

CXX="${CXX:-g++}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
LIBFABRIC_HOME="${LIBFABRIC_HOME:-/opt/amazon/efa}"

CXXFLAGS="${CXXFLAGS:-} -O2 -std=c++17 -fPIC"
INCS="-I${LIBFABRIC_HOME}/include -I${CUDA_HOME}/include"
LIBS="-L${LIBFABRIC_HOME}/lib -lfabric -L${CUDA_HOME}/lib64 -lcudart"

echo "Compiling gda_probe..."
${CXX} ${CXXFLAGS} ${INCS} -o "${OUT_DIR}/gda_probe" \
  "$(dirname "$0")/../test/gda_probe_standalone.cpp" ${LIBS}

echo "Built: ${OUT_DIR}/gda_probe"
