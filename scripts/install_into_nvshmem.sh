#!/usr/bin/env bash
set -euo pipefail

# Inject EFA GDA scaffolding into an NVSHMEM source tree and rebuild the libfabric transport.
# Usage:
#   NVSHMEM_SRC=/path/to/nvshmem_src \
#   LIBFABRIC_HOME=/opt/amazon/efa \
#   CUDA_HOME=/usr/local/cuda \
#   ./scripts/install_into_nvshmem.sh
#
# This script:
#   1) Copies include/ and src/ files under transports/libfabric/gda/
#   2) Adds an #include to the main libfabric transport TU (if matched)
#   3) Adds a call to nvshmemt_efa_gda_init_on_ep(...) after ep enable
#   4) Adds a call to nvshmemt_efa_gda_fini(...) in finalize
#   5) Extends CMakeLists to compile the .cu and .cpp
#   6) Rebuilds with Ninja
#
# If a sed patch fails (pattern not found), we print manual instructions.

: "${NVSHMEM_SRC:?Set NVSHMEM_SRC to NVSHMEM source root}"
: "${CUDA_HOME:=/usr/local/cuda}"

GDA_REL_DIR="transports/libfabric/gda"
DST_DIR="${NVSHMEM_SRC}/${GDA_REL_DIR}"
mkdir -p "${DST_DIR}"
cp -v "$(dirname "$0")/../include/transport_nvshmem_efa_gda.h" "${DST_DIR}/"
cp -v "$(dirname "$0")/../src/transport_nvshmem_efa_gda.cpp" "${DST_DIR}/"
cp -v "$(dirname "$0")/../src/transport_nvshmem_efa_gda.cu" "${DST_DIR}/"

# 2) Try to add include to a known TU; search for libfabric transport implementation.
TRANSPORT_CXX=$(grep -RIl "nvshmemt_libfabric_connect_endpoints" "${NVSHMEM_SRC}" || true)
if [[ -z "${TRANSPORT_CXX}" ]]; then
  echo "WARN: Could not locate libfabric transport TU automatically."
  echo "Add this include near the other transport includes:"
  echo "  #include \"${GDA_REL_DIR}/transport_nvshmem_efa_gda.h\""
else
  if ! grep -q "${GDA_REL_DIR}/transport_nvshmem_efa_gda.h" "${TRANSPORT_CXX}"; then
    sed -i "1i #include \"${GDA_REL_DIR}/transport_nvshmem_efa_gda.h\"" "${TRANSPORT_CXX}" || true
  fi
  echo "Patched include into: ${TRANSPORT_CXX}"
fi

# 3) Call nvshmemt_efa_gda_init_on_ep(...) after endpoints are enabled.
# We try to locate where fi_enable(ep) is called in connect_endpoints.
CONNECT_FILE="${TRANSPORT_CXX}"
if [[ -n "${CONNECT_FILE}" ]]; then
  if ! grep -q "nvshmemt_efa_gda_init_on_ep" "${CONNECT_FILE}"; then
    sed -i '/fi_enable(.*endpoint.*);/a \
      /* EFA GDA init: domain + one EP */\
      nvshmemt_efa_gda_init_on_ep(state->domain, state->eps[0].endpoint);' "${CONNECT_FILE}" || true
  fi
  echo "Patched init call into: ${CONNECT_FILE} (if pattern matched)"
else
  echo "WARN: Could not patch init site automatically. Insert call manually after fi_enable(ep):"
  echo "  nvshmemt_efa_gda_init_on_ep(state->domain, state->eps[0].endpoint);"
fi

# 4) Call fini in finalize
FINALIZE_CXX=$(grep -RIl "nvshmemt_libfabric_finalize" "${NVSHMEM_SRC}" || true)
if [[ -n "${FINALIZE_CXX}" ]]; then
  if ! grep -q "nvshmemt_efa_gda_fini" "${FINALIZE_CXX}"; then
    sed -i '/nvshmemt_libfabric_finalize/,$!b;/{/,/}/ s/}/  nvshmemt_efa_gda_fini();\n}/' "${FINALIZE_CXX}" || true
  fi
  echo "Patched fini into: ${FINALIZE_CXX} (if pattern matched)"
else
  echo "WARN: Could not patch finalize automatically. Add before tearing down domain:"
  echo "  nvshmemt_efa_gda_fini();"
fi

# 5) Update CMake to compile our files
CMAKE_FILE=$(grep -RIl "add_library" "${NVSHMEM_SRC}/transports/libfabric" | head -n1 || true)
if [[ -n "${CMAKE_FILE}" ]]; then
  if ! grep -q "transport_nvshmem_efa_gda.cpp" "${CMAKE_FILE}"; then
    cat >> "${CMAKE_FILE}" <<'EOF'

# --- EFA GDA scaffolding ---
add_library(nvshmem_efa_gda_objs OBJECT
  gda/transport_nvshmem_efa_gda.cpp
  gda/transport_nvshmem_efa_gda.cu
)
set_property(TARGET nvshmem_efa_gda_objs PROPERTY CUDA_STANDARD 17)
set_property(TARGET nvshmem_efa_gda_objs PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(nvshmem_efa_gda_objs PRIVATE ${CUDA_HOME}/include)
# Link these objects into the primary libfabric transport target (if named nvshmem_transport_libfabric)
# If your target is named differently, append this OBJECT library accordingly:
# target_sources(nvshmem_transport_libfabric PRIVATE $<TARGET_OBJECTS:nvshmem_efa_gda_objs>)
EOF
  fi
  echo "Appended CMake fragments into: ${CMAKE_FILE}"
else
  echo "WARN: Could not find CMakeLists under transports/libfabric automatically."
  echo "Add the following to that CMakeLists.txt:"
  cat <<'EOF'
add_library(nvshmem_efa_gda_objs OBJECT
  gda/transport_nvshmem_efa_gda.cpp
  gda/transport_nvshmem_efa_gda.cu
)
set_property(TARGET nvshmem_efa_gda_objs PROPERTY CUDA_STANDARD 17)
set_property(TARGET nvshmem_efa_gda_objs PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(nvshmem_efa_gda_objs PRIVATE ${CUDA_HOME}/include)
# Append OBJECTs to your main transport target:
# target_sources(nvshmem_transport_libfabric PRIVATE $<TARGET_OBJECTS:nvshmem_efa_gda_objs>)
EOF
fi

# 6) Rebuild using the top-level NVSHMEM CMake/Ninja if present
if [[ -d "${NVSHMEM_SRC}/build" ]]; then
  cmake --build "${NVSHMEM_SRC}/build" --target install -j
else
  echo "INFO: No NVSHMEM build dir found. Build manually, e.g.:"
  echo "  cmake -S ${NVSHMEM_SRC} -B ${NVSHMEM_SRC}/build -G Ninja ..."
  echo "  cmake --build ${NVSHMEM_SRC}/build --target install -j"
fi

echo "Done."
