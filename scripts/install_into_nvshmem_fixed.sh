#!/usr/bin/env bash
set -euo pipefail

# Fixed version of install_into_nvshmem.sh
# Works with NVSHMEM 2.x source tree structure
#
# Usage:
#   NVSHMEM_SRC=/path/to/nvshmem_src \
#   LIBFABRIC_HOME=/opt/amazon/efa \
#   CUDA_HOME=/usr/local/cuda \
#   ./install_into_nvshmem_fixed.sh

: "${NVSHMEM_SRC:?Set NVSHMEM_SRC to NVSHMEM source root}"
: "${CUDA_HOME:=/usr/local/cuda}"

echo "=========================================="
echo "EFA GDA Integration (Fixed Version)"
echo "=========================================="
echo ""
echo "NVSHMEM_SRC: ${NVSHMEM_SRC}"
echo "CUDA_HOME: ${CUDA_HOME}"
echo ""

# Step 1: Locate source files
EFA_GDA_SRC="$(dirname "$0")/nvshmem-efa-gda"
if [[ ! -d "${EFA_GDA_SRC}" ]]; then
  # Try alternative location
  EFA_GDA_SRC="/home/ubuntu/nvshmem-efa-gda"
fi

if [[ ! -d "${EFA_GDA_SRC}" ]]; then
  echo "ERROR: Cannot find nvshmem-efa-gda source directory"
  echo "Expected at: $(dirname "$0")/nvshmem-efa-gda"
  echo "Or at: /home/ubuntu/nvshmem-efa-gda"
  exit 1
fi

echo "[1/7] Copying GDA files from ${EFA_GDA_SRC}..."
GDA_REL_DIR="transports/libfabric/gda"
DST_DIR="${NVSHMEM_SRC}/${GDA_REL_DIR}"
mkdir -p "${DST_DIR}"

cp -v "${EFA_GDA_SRC}/include/transport_nvshmem_efa_gda.h" "${DST_DIR}/"
cp -v "${EFA_GDA_SRC}/src/transport_nvshmem_efa_gda.cpp" "${DST_DIR}/"
cp -v "${EFA_GDA_SRC}/src/transport_nvshmem_efa_gda.cu" "${DST_DIR}/"

if [[ ! -f "${DST_DIR}/transport_nvshmem_efa_gda.h" ]]; then
  echo "ERROR: Could not copy GDA files."
  exit 1
fi
echo ""

# Step 1.5: Fix include paths and copy third_party
echo "[2/7] Fixing include paths in copied files..."

# Fix transport_nvshmem_efa_gda.cpp: "../include/..." -> local include
sed -i 's|#include "../include/transport_nvshmem_efa_gda.h"|#include "transport_nvshmem_efa_gda.h"|g' \
    "${DST_DIR}/transport_nvshmem_efa_gda.cpp"
echo "  ✓ Fixed .cpp include"

# Fix transport_nvshmem_efa_gda.cu: "../include/..." -> local include
sed -i 's|#include "../include/transport_nvshmem_efa_gda.h"|#include "transport_nvshmem_efa_gda.h"|g' \
    "${DST_DIR}/transport_nvshmem_efa_gda.cu"
echo "  ✓ Fixed .cu include"

# Copy third_party directory
THIRD_PARTY_DST="${NVSHMEM_SRC}/transports/libfabric/third_party"
mkdir -p "${THIRD_PARTY_DST}"
cp -r "${EFA_GDA_SRC}/third_party/efa_gda" "${THIRD_PARTY_DST}/"
echo "  ✓ Copied third_party/efa_gda"
echo ""

# Step 2: Find the libfabric transport source file
echo "[3/7] Locating libfabric transport source..."
# Try multiple search patterns for different NVSHMEM versions
TRANSPORT_CXX=""

# Method 1: Search for specific functions
for pattern in "nvshmemt_libfabric_connect_endpoints" "fi_fabric" "fi_domain"; do
  FOUND=$(find "${NVSHMEM_SRC}" -name "*.cpp" -type f -exec grep -l "${pattern}" {} \; 2>/dev/null | grep -i libfabric | head -1 || true)
  if [[ -n "${FOUND}" ]]; then
    TRANSPORT_CXX="${FOUND}"
    break
  fi
done

# Method 2: Known path patterns
if [[ -z "${TRANSPORT_CXX}" ]]; then
  for path in \
    "${NVSHMEM_SRC}/src/modules/transport/libfabric/libfabric.cpp" \
    "${NVSHMEM_SRC}/transports/libfabric/transport.cpp" \
    "${NVSHMEM_SRC}/src/transport/libfabric/libfabric.cpp"; do
    if [[ -f "${path}" ]]; then
      TRANSPORT_CXX="${path}"
      break
    fi
  done
fi

if [[ -z "${TRANSPORT_CXX}" ]]; then
  echo "ERROR: Could not locate libfabric transport source file."
  echo "Manually add the following to your libfabric transport .cpp file:"
  echo "  #include \"../../../${GDA_REL_DIR}/transport_nvshmem_efa_gda.h\""
  exit 1
fi

echo "  Found: ${TRANSPORT_CXX}"
echo ""

# Step 3: Add include
echo "[4/7] Adding header include..."
if grep -q "transport_nvshmem_efa_gda.h" "${TRANSPORT_CXX}"; then
  echo "  Already added, skipping"
else
  # Calculate relative path from transport file to gda directory
  TRANSPORT_DIR=$(dirname "${TRANSPORT_CXX}")
  REL_PATH=$(realpath --relative-to="${TRANSPORT_DIR}" "${DST_DIR}" || echo "../../../${GDA_REL_DIR}")

  # Add include after other includes (look for last #include line)
  LAST_INCLUDE_LINE=$(grep -n "^#include" "${TRANSPORT_CXX}" | tail -1 | cut -d: -f1)
  if [[ -n "${LAST_INCLUDE_LINE}" ]]; then
    sed -i "${LAST_INCLUDE_LINE}a #include \"${REL_PATH}/transport_nvshmem_efa_gda.h\"" "${TRANSPORT_CXX}"
    echo "  Added at line $((LAST_INCLUDE_LINE + 1))"
  else
    # Fallback: add at top
    sed -i "1i #include \"${REL_PATH}/transport_nvshmem_efa_gda.h\"" "${TRANSPORT_CXX}"
    echo "  Added at top of file"
  fi
fi
echo ""

# Step 4: Add init call after fi_enable
echo "[5/7] Adding GDA init call..."
if grep -q "nvshmemt_efa_gda_init_on_ep" "${TRANSPORT_CXX}"; then
  echo "  Already added, skipping"
else
  # Find fi_enable line
  FI_ENABLE_LINE=$(grep -n "fi_enable.*endpoint" "${TRANSPORT_CXX}" | head -1 | cut -d: -f1)

  if [[ -n "${FI_ENABLE_LINE}" ]]; then
    # Find the next line after error checking (usually 2-3 lines after fi_enable)
    INSERT_LINE=$((FI_ENABLE_LINE + 3))

    # Insert the init call
    sed -i "${INSERT_LINE}a \\
        /* EFA GDA: Map Send Queue and Doorbell to GPU */\\
        if (i == 0 && state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {\\
            status = nvshmemt_efa_gda_init_on_ep(state->domain, state->eps[0].endpoint);\\
            if (status != 0) {\\
                fprintf(stderr, \"WARN: EFA GDA init failed: %d. Continuing with standard transport.\\\\n\", status);\\
            } else {\\
                fprintf(stderr, \"INFO: EFA GDA initialized successfully.\\\\n\");\\
            }\\
        }" "${TRANSPORT_CXX}"

    echo "  Added at line ${INSERT_LINE}"
  else
    echo "  WARNING: Could not find fi_enable call. Add manually:"
    echo "    nvshmemt_efa_gda_init_on_ep(state->domain, state->eps[0].endpoint);"
  fi
fi
echo ""

# Step 5: Add fini call in finalize
echo "[6/7] Adding GDA fini call..."
if grep -q "nvshmemt_efa_gda_fini" "${TRANSPORT_CXX}"; then
  echo "  Already added, skipping"
else
  # Find finalize function
  FINALIZE_LINE=$(grep -n "nvshmemt_libfabric_finalize" "${TRANSPORT_CXX}" | grep -v "//" | head -1 | cut -d: -f1)

  if [[ -n "${FINALIZE_LINE}" ]]; then
    # Add fini call at start of function (after opening brace and variable declarations)
    INSERT_LINE=$((FINALIZE_LINE + 8))

    sed -i "${INSERT_LINE}i \\
    /* EFA GDA: Cleanup */\\
    nvshmemt_efa_gda_fini();" "${TRANSPORT_CXX}"

    echo "  Added at line ${INSERT_LINE}"
  else
    echo "  WARNING: Could not find finalize function. Add manually:"
    echo "    nvshmemt_efa_gda_fini();"
  fi
fi
echo ""

# Step 6: Update CMakeLists.txt
echo "[7/7] Updating CMakeLists.txt..."
CMAKE_FILE=""

# Find CMakeLists.txt
for path in \
  "${NVSHMEM_SRC}/src/CMakeLists.txt" \
  "${NVSHMEM_SRC}/transports/libfabric/CMakeLists.txt" \
  "${NVSHMEM_SRC}/CMakeLists.txt"; do
  if [[ -f "${path}" ]] && grep -q "nvshmem_transport_libfabric" "${path}"; then
    CMAKE_FILE="${path}"
    break
  fi
done

if [[ -z "${CMAKE_FILE}" ]]; then
  echo "  WARNING: Could not find CMakeLists.txt with libfabric target"
  echo "  Manually add GDA files to your CMake configuration"
else
  echo "  Found: ${CMAKE_FILE}"

  if grep -q "transport_nvshmem_efa_gda.cpp" "${CMAKE_FILE}"; then
    echo "  Already configured, skipping"
  else
    # Find target_sources for libfabric
    TARGET_LINE=$(grep -n "target_sources(nvshmem_transport_libfabric" "${CMAKE_FILE}" | head -1 | cut -d: -f1)

    if [[ -n "${TARGET_LINE}" ]]; then
      # Add our files right after target_sources line
      INSERT_LINE=$((TARGET_LINE + 1))

      # Calculate relative path from CMakeLists.txt to gda directory
      CMAKE_DIR=$(dirname "${CMAKE_FILE}")
      REL_PATH=$(realpath --relative-to="${CMAKE_DIR}" "${DST_DIR}" || echo "../${GDA_REL_DIR}")

      sed -i "${INSERT_LINE}i \\
                 ${REL_PATH}/transport_nvshmem_efa_gda.cpp\\
                 ${REL_PATH}/transport_nvshmem_efa_gda.cu" "${CMAKE_FILE}"

      echo "  Added GDA sources at line ${INSERT_LINE}"

      # Add CUDA properties if not present
      if ! grep -q "CUDA_STANDARD.*nvshmem_transport_libfabric" "${CMAKE_FILE}"; then
        # Find ACTIVE_TRANSPORTS line
        ACTIVE_LINE=$(grep -n "ACTIVE_TRANSPORTS.*nvshmem_transport_libfabric" "${CMAKE_FILE}" | head -1 | cut -d: -f1)
        if [[ -n "${ACTIVE_LINE}" ]]; then
          sed -i "${ACTIVE_LINE}a \\
  # EFA GDA: Enable CUDA compilation\\
  set_property(TARGET nvshmem_transport_libfabric PROPERTY CUDA_STANDARD 17)\\
  set_property(TARGET nvshmem_transport_libfabric PROPERTY CUDA_ARCHITECTURES native)" "${CMAKE_FILE}"
          echo "  Added CUDA properties"
        fi
      fi
    else
      echo "  WARNING: Could not find target_sources line"
    fi
  fi
fi
echo ""

# Step 7: Summary and next steps
echo "=========================================="
echo "Integration Complete!"
echo "=========================================="
echo ""
echo "Modified files:"
echo "  - ${TRANSPORT_CXX}"
if [[ -n "${CMAKE_FILE}" ]]; then
  echo "  - ${CMAKE_FILE}"
fi
echo ""
echo "Next steps:"
echo "  1. Clean and rebuild:"
echo "     cd ${NVSHMEM_SRC}/build"
echo "     rm -rf *"
echo "     cmake .. -DCUDA_HOME=${CUDA_HOME} -DLIBFABRIC_HOME=\${LIBFABRIC_HOME:-/opt/amazon/efa}"
echo "     make -j\$(nproc)"
echo ""
echo "  2. Verify compilation:"
echo "     grep 'transport_nvshmem_efa_gda' build.log"
echo ""
echo "  3. Test with:"
echo "     export NVSHMEM_REMOTE_TRANSPORT=libfabric"
echo "     export NVSHMEM_LIBFABRIC_PROVIDER=efa"
echo "     export FI_EFA_USE_DEVICE_RDMA=1"
echo ""
