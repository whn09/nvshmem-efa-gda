// SPDX-License-Identifier: BSD-3-Clause
// Host-side initialization for NVSHMEM + EFA GPUDirect Async
// Based on official libfabric fabtests implementation:
// https://github.com/ofiwg/libfabric/blob/main/fabtests/prov/efa/src/efa_gda.c

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
extern __device__ uint32_t  __nvshmem_efa_sq_phase;
extern __device__ uint32_t  __nvshmem_efa_sq_mask;

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

    // Queue properties
    uint32_t entry_size = 0;
    uint32_t num_entries = 0;
};

static efa_gda_state_t g_state;
static std::once_flag  g_init_once;

// Helper macro for CUDA error checking
#define CUDA_TRY(call)                                                  \
    do {                                                                \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess) {                                        \
            fprintf(stderr, "[EFA-GDA] CUDA error %d: %s at %s:%d\n",   \
                    (int)_e, cudaGetErrorString(_e), __FILE__, __LINE__);\
            return -1;                                                  \
        }                                                               \
    } while (0)

// Copy a host pointer to a device symbol (pointer variable)
template <typename T>
static int publish_ptr_symbol(const void* symbol, T* host_ptr) {
    return (int)cudaMemcpyToSymbol(symbol, &host_ptr, sizeof(T*));
}

template <typename T>
static int publish_val_symbol(const void* symbol, const T& value) {
    return (int)cudaMemcpyToSymbol(symbol, &value, sizeof(T));
}

/**
 * Map host memory for device access using CUDA Host Register
 * Based on efa_gda.c implementation using cuMemHostRegister
 */
static int map_for_device(void* host_ptr, size_t bytes, void** dev_ptr_out, bool* registered_flag) {
    // Register host memory with CUDA for device mapping
    // Use cudaHostRegisterIoMemory flag for EFA queue buffers
    cudaError_t ce = cudaHostRegister(host_ptr, bytes,
                                       cudaHostRegisterIoMemory | cudaHostRegisterMapped);
    if (ce == cudaSuccess) {
        *registered_flag = true;
    } else if (ce == cudaErrorHostMemoryAlreadyRegistered ||
               ce == cudaErrorNotSupported) {
        // Non-fatal: memory may already be registered by libfabric
        *registered_flag = false;
        cudaGetLastError(); // clear error
        fprintf(stderr, "[EFA-GDA] cudaHostRegister: %s (continuing)\n", cudaGetErrorString(ce));
    } else {
        fprintf(stderr, "[EFA-GDA] cudaHostRegister failed: %s\n", cudaGetErrorString(ce));
        return -1;
    }

    // Get device pointer mapping to host memory
    void* dev_ptr = nullptr;
    ce = cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0);
    if (ce != cudaSuccess) {
        fprintf(stderr, "[EFA-GDA] cudaHostGetDevicePointer failed: %s\n", cudaGetErrorString(ce));
        if (*registered_flag) cudaHostUnregister(host_ptr);
        return -1;
    }

    *dev_ptr_out = dev_ptr;
    fprintf(stderr, "[EFA-GDA] Mapped host %p -> device %p (%zu bytes)\n",
            host_ptr, dev_ptr, bytes);
    return 0;
}

/**
 * Initialize EFA GDA for a given endpoint
 * Based on efa_gda.c setup_gda_and_cuda()
 */
int nvshmemt_efa_gda_init_on_ep(struct fid_domain* domain, struct fid_ep* ep) {
    int rc = 0;
    std::call_once(g_init_once, [&]() {
        memset(&g_state, 0, sizeof(g_state));
    });

    // 1. Check environment variable for GPU direct RDMA
    const char* v = std::getenv("FI_EFA_USE_DEVICE_RDMA");
    if (!v || strcmp(v, "1") != 0) {
        fprintf(stderr,
            "[EFA-GDA] WARNING: FI_EFA_USE_DEVICE_RDMA is not set to 1. "
            "NIC may not fetch GPU memory.\n");
    }

    // 2. Open EFA GDA operations on the domain
    rc = fi_open_ops(&domain->fid, FI_EFA_GDA_OPS, 0, (void**)&g_state.ops, NULL);
    if (rc) {
        fprintf(stderr, "[EFA-GDA] fi_open_ops(FI_EFA_GDA_OPS) failed: %d\n", rc);
        return rc;
    }

    // 3. Query GDA support
    rc = g_state.ops->query_gda_support(domain);
    if (rc) {
        fprintf(stderr, "[EFA-GDA] query_gda_support failed: %d (GDA not supported)\n", rc);
        return rc;
    }
    fprintf(stderr, "[EFA-GDA] GDA support confirmed\n");

    // 4. Query QP Work Queue attributes (send & recv)
    memset(&g_state.sq_attr, 0, sizeof(g_state.sq_attr));
    memset(&g_state.rq_attr, 0, sizeof(g_state.rq_attr));
    rc = g_state.ops->query_qp_wqs(ep, &g_state.sq_attr, &g_state.rq_attr);
    if (rc) {
        fprintf(stderr, "[EFA-GDA] query_qp_wqs failed: %d\n", rc);
        return rc;
    }

    // Extract queue attributes
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
        fprintf(stderr, "[EFA-GDA] Invalid SQ attributes.\n");
        return -1;
    }

    // 5. Map SQ buffer for device access
    if (map_for_device(g_state.sq_buf_host, g_state.sq_bytes,
                       (void**)&g_state.sq_buf_dev, &g_state.sq_buf_registered)) {
        fprintf(stderr, "[EFA-GDA] map_for_device(SQ buffer) failed.\n");
        return -1;
    }

    // 6. Map doorbell for device access
    if (g_state.db_host) {
        if (map_for_device(g_state.db_host, sizeof(uint32_t),
                           (void**)&g_state.db_dev, &g_state.db_registered)) {
            fprintf(stderr, "[EFA-GDA] map_for_device(doorbell) failed; continuing without device doorbell.\n");
            g_state.db_dev = nullptr;
        }
    } else {
        fprintf(stderr, "[EFA-GDA] No doorbell provided by libfabric\n");
    }

    // 7. Publish device symbols
    rc = publish_ptr_symbol((const void*)&__nvshmem_efa_sq_buf, g_state.sq_buf_dev);
    if (rc) { fprintf(stderr, "[EFA-GDA] publish sq_buf_dev failed (%d)\n", rc); return -1; }

    rc = publish_ptr_symbol((const void*)&__nvshmem_efa_sq_db, g_state.db_dev);
    if (rc) { fprintf(stderr, "[EFA-GDA] publish db_dev failed (%d)\n", rc); /* not fatal */ }

    rc = publish_val_symbol((const void*)&__nvshmem_efa_sq_stride, g_state.entry_size);
    if (rc) { fprintf(stderr, "[EFA-GDA] publish stride failed (%d)\n", rc); return -1; }

    rc = publish_val_symbol((const void*)&__nvshmem_efa_sq_size, g_state.num_entries);
    if (rc) { fprintf(stderr, "[EFA-GDA] publish size failed (%d)\n", rc); return -1; }

    // 8. Initialize queue mask (for efficient modulo operation)
    uint32_t queue_mask = g_state.num_entries - 1;
    rc = publish_val_symbol((const void*)&__nvshmem_efa_sq_mask, queue_mask);
    if (rc) { fprintf(stderr, "[EFA-GDA] publish mask failed (%d)\n", rc); return -1; }

    // 9. Reset device tail and phase to initial values
    uint32_t zero = 0;
    uint32_t phase_init = 1;  // Initial phase is 1

    rc = (int)cudaMemcpyToSymbol(__nvshmem_efa_sq_tail, &zero, sizeof(uint32_t));
    if (rc) { fprintf(stderr, "[EFA-GDA] publish tail failed (%d)\n", rc); return -1; }

    rc = (int)cudaMemcpyToSymbol(__nvshmem_efa_sq_phase, &phase_init, sizeof(uint32_t));
    if (rc) { fprintf(stderr, "[EFA-GDA] publish phase failed (%d)\n", rc); return -1; }

    fprintf(stderr, "[EFA-GDA] Successfully initialized and published SQ mapping to device.\n");
    fprintf(stderr, "[EFA-GDA]   SQ buffer: %p (device: %p)\n", g_state.sq_buf_host, g_state.sq_buf_dev);
    fprintf(stderr, "[EFA-GDA]   Doorbell:  %p (device: %p)\n", g_state.db_host, g_state.db_dev);
    fprintf(stderr, "[EFA-GDA]   Queue:     %u entries x %u bytes, mask=0x%x\n",
            g_state.num_entries, g_state.entry_size, queue_mask);

    return 0;
}

/**
 * Cleanup EFA GDA resources
 */
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
    cudaMemcpyToSymbol(__nvshmem_efa_sq_phase,  &zero, sizeof(zero));
    cudaMemcpyToSymbol(__nvshmem_efa_sq_mask,   &zero, sizeof(zero));

    // Unregister host memory if we registered it
    if (g_state.sq_buf_host && g_state.sq_buf_registered) {
        cudaHostUnregister(g_state.sq_buf_host);
    }
    if (g_state.db_host && g_state.db_registered) {
        cudaHostUnregister(g_state.db_host);
    }

    memset(&g_state, 0, sizeof(g_state));
    fprintf(stderr, "[EFA-GDA] Finalized.\n");
}
