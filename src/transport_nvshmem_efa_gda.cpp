// SPDX-License-Identifier: BSD-3-Clause
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <atomic>

#include <rdma/fi_ext_efa.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "../include/transport_nvshmem_efa_gda.h"

// Device symbols defined in the .cu TU
extern __device__ uint8_t*  __nvshmem_efa_sq_buf;
extern __device__ uint32_t* __nvshmem_efa_sq_db;
extern __device__ uint32_t  __nvshmem_efa_sq_stride;
extern __device__ uint32_t  __nvshmem_efa_sq_size;
extern __device__ uint32_t  __nvshmem_efa_sq_tail;

// Host-side shadow state for EFA GDA
struct efa_gda_state_t {
    const struct fi_efa_ops_gda* ops = nullptr;
    struct fi_efa_wq_attr sq_attr{};
    struct fi_efa_wq_attr rq_attr{};

    void*    sq_buf_host = nullptr;
    size_t   sq_bytes = 0;
    void*    db_host = nullptr;

    // Device-mapped pointers
    uint8_t*  sq_buf_dev = nullptr;
    uint32_t* db_dev     = nullptr;

    // Flags
    bool sq_buf_registered = false;
    bool db_registered     = false;

    // queue properties
    uint32_t entry_size = 0;
    uint32_t num_entries = 0;
};
static efa_gda_state_t g_state;
static std::once_flag  g_init_once;

// Small helper for CUDA error checking
#define CUDA_TRY(call)                                                  \
    do {                                                                \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess) {                                        \
            fprintf(stderr, "[EFA-GDA] CUDA error %d: %s at %s:%d\n",   \
                    (int)_e, cudaGetErrorString(_e), __FILE__, __LINE__);\
            return -1;                                                  \
        }                                                               \
    } while (0)

// Copy a host pointer to a device symbol (pointer variable).
template <typename T>
static int publish_ptr_symbol(const void* symbol, T* host_ptr) {
    return (int)cudaMemcpyToSymbol(symbol, &host_ptr, sizeof(T*));
}

template <typename T>
static int publish_val_symbol(const void* symbol, const T& value) {
    return (int)cudaMemcpyToSymbol(symbol, &value, sizeof(T));
}

static int map_for_device(void* host_ptr, size_t bytes, void** dev_ptr_out, bool* registered_flag) {
    // Try to register (best-effort; some providers already pin/map)
    cudaError_t ce = cudaHostRegister(host_ptr, bytes, cudaHostRegisterPortable);
    if (ce == cudaSuccess) {
        *registered_flag = true;
    } else if (ce == cudaErrorHostMemoryAlreadyRegistered ||
               ce == cudaErrorNotSupported) {
        // Non-fatal: continue; try to get device pointer directly
        *registered_flag = false;
        cudaGetLastError(); // clear
    } else {
        fprintf(stderr, "[EFA-GDA] cudaHostRegister failed: %s\n", cudaGetErrorString(ce));
        return -1;
    }

    // Get a device pointer mapping to host memory
    void* dev_ptr = nullptr;
    ce = cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0);
    if (ce != cudaSuccess) {
        fprintf(stderr, "[EFA-GDA] cudaHostGetDevicePointer failed: %s\n", cudaGetErrorString(ce));
        if (*registered_flag) cudaHostUnregister(host_ptr);
        return -1;
    }
    *dev_ptr_out = dev_ptr;
    return 0;
}

int nvshmemt_efa_gda_init_on_ep(struct fid_domain* domain, struct fid_ep* ep) {
    int rc = 0;
    std::call_once(g_init_once, [&]() {
        memset(&g_state, 0, sizeof(g_state));
    });

    // Sanity: check env needed for GPU direct
    const char* v = std::getenv("FI_EFA_USE_DEVICE_RDMA");
    if (!v || strcmp(v, "1") != 0) {
        fprintf(stderr,
            "[EFA-GDA] WARNING: FI_EFA_USE_DEVICE_RDMA is not set to 1. "
            "NIC may not fetch GPU memory.\n");
    }

    // 1) Open ops for FI_EFA_GDA_OPS on the domain
    rc = fi_open_ops(&domain->fid, FI_EFA_GDA_OPS, 0, (void**)&g_state.ops, NULL);
    if (rc) {
        fprintf(stderr, "[EFA-GDA] fi_open_ops(FI_EFA_GDA_OPS) failed: %d\n", rc);
        return rc;
    }

    // 2) Query QP WQ attributes (send & recv)
    memset(&g_state.sq_attr, 0, sizeof(g_state.sq_attr));
    memset(&g_state.rq_attr, 0, sizeof(g_state.rq_attr));
    rc = g_state.ops->query_qp_wqs(ep, &g_state.sq_attr, &g_state.rq_attr);
    if (rc) {
        fprintf(stderr, "[EFA-GDA] query_qp_wqs failed: %d\n", rc);
        return rc;
    }

    // Expect these fields in fi_efa_wq_attr. If names differ in your libfabric,
    // adapt here.
    g_state.sq_buf_host = g_state.sq_attr.buffer;
    g_state.db_host     = g_state.sq_attr.doorbell;
    g_state.entry_size  = (uint32_t)g_state.sq_attr.entry_size;
    g_state.num_entries = (uint32_t)g_state.sq_attr.num_entries;
    g_state.sq_bytes    = (size_t)g_state.entry_size * (size_t)g_state.num_entries;

    fprintf(stderr,
        "[EFA-GDA] SQ: buf=%p bytes=%zu entry_size=%u num_entries=%u db=%p\n",
        g_state.sq_buf_host, g_state.sq_bytes, g_state.entry_size,
        g_state.num_entries, g_state.db_host);

    if (!g_state.sq_buf_host || !g_state.entry_size || !g_state.num_entries) {
        fprintf(stderr, "[EFA-GDA] invalid SQ attributes.\n");
        return -1;
    }

    // 3) Map SQ buffer and doorbell for device access
    if (map_for_device(g_state.sq_buf_host, g_state.sq_bytes,
                       (void**)&g_state.sq_buf_dev, &g_state.sq_buf_registered)) {
        fprintf(stderr, "[EFA-GDA] map_for_device(SQ buffer) failed.\n");
        return -1;
    }

    if (g_state.db_host) {
        if (map_for_device(g_state.db_host, sizeof(uint32_t),
                           (void**)&g_state.db_dev, &g_state.db_registered)) {
            fprintf(stderr, "[EFA-GDA] map_for_device(doorbell) failed; continuing without device doorbell.\n");
            g_state.db_dev = nullptr;
        }
    }

    // 4) Publish device symbols
    rc = publish_ptr_symbol((const void*)&__nvshmem_efa_sq_buf, g_state.sq_buf_dev);
    if (rc) { fprintf(stderr, "[EFA-GDA] publish sq_buf_dev failed (%d)\n", rc); return -1; }
    rc = publish_ptr_symbol((const void*)&__nvshmem_efa_sq_db,  g_state.db_dev);
    if (rc) { fprintf(stderr, "[EFA-GDA] publish db_dev failed (%d)\n", rc); /* not fatal */ }
    rc = publish_val_symbol((const void*)&__nvshmem_efa_sq_stride, g_state.entry_size);
    if (rc) { fprintf(stderr, "[EFA-GDA] publish stride failed (%d)\n", rc); return -1; }
    rc = publish_val_symbol((const void*)&__nvshmem_efa_sq_size, g_state.num_entries);
    if (rc) { fprintf(stderr, "[EFA-GDA] publish size failed (%d)\n", rc); return -1; }

    // Reset device tail to 0
    uint32_t zero = 0;
    rc = (int)cudaMemcpyToSymbol(__nvshmem_efa_sq_tail, &zero, sizeof(uint32_t));
    if (rc) { fprintf(stderr, "[EFA-GDA] publish tail failed (%d)\n", rc); return -1; }

    fprintf(stderr, "[EFA-GDA] Initialized and published SQ mapping to device.\n");
    return 0;
}

void nvshmemt_efa_gda_fini(void) {
    // Clear device symbols
    uint8_t*  nul8  = nullptr;
    uint32_t* nul32 = nullptr;
    uint32_t  zero  = 0;
    cudaMemcpyToSymbol(__nvshmem_efa_sq_buf, &nul8,  sizeof(nul8));
    cudaMemcpyToSymbol(__nvshmem_efa_sq_db,  &nul32, sizeof(nul32));
    cudaMemcpyToSymbol(__nvshmem_efa_sq_stride, &zero, sizeof(zero));
    cudaMemcpyToSymbol(__nvshmem_efa_sq_size,   &zero, sizeof(zero));
    cudaMemcpyToSymbol(__nvshmem_efa_sq_tail,   &zero, sizeof(zero));

    // Unregister if we registered
    if (g_state.sq_buf_host && g_state.sq_buf_registered) {
        cudaHostUnregister(g_state.sq_buf_host);
    }
    if (g_state.db_host && g_state.db_registered) {
        cudaHostUnregister(g_state.db_host);
    }

    memset(&g_state, 0, sizeof(g_state));
    fprintf(stderr, "[EFA-GDA] Finalized.\n");
}
