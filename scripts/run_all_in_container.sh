#!/usr/bin/env bash
set -euo pipefail

ZIP_PATH="${1:-/root/nvshmem-efa-gda.zip}"
WORK_DIR="/root/nvshmem-efa-gda"
mkdir -p "${WORK_DIR}"

echo "==> Unzipping ${ZIP_PATH} into ${WORK_DIR}"
unzip -o "${ZIP_PATH}" -d "/root"

echo "==> Building standalone probe"
"${WORK_DIR}/scripts/build_standalone_probe.sh" "/workspace"
echo "==> Running probe"
/workspace/gda_probe || true

echo "==> Fetching NVSHMEM source and installing GDA scaffolding"
NVSHMEM_VER="${NVSHMEM_VER:-3.2.5-1}"
cd /opt
mkdir -p /tmp/nvshmem-src
cd /tmp/nvshmem-src
curl -L "https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_${NVSHMEM_VER}.txz" -o "nvshmem_src_${NVSHMEM_VER}.txz"
tar -xf "nvshmem_src_${NVSHMEM_VER}.txz"
cd nvshmem_src

export NVSHMEM_SRC="$(pwd)"
"${WORK_DIR}/scripts/install_into_nvshmem.sh"

echo ""
echo "==> Export these when running with the rebuilt NVSHMEM:"
cat <<'ENVVARS'
export NVSHMEM_HOME=/opt/nvshmem
export PATH=$NVSHMEM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export NVSHMEM_REMOTE_TRANSPORT=libfabric
export NVSHMEM_LIBFABRIC_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
ENVVARS

echo "All done."
