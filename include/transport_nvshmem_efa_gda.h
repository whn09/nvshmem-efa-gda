// SPDX-License-Identifier: BSD-3-Clause
// NVSHMEM + EFA GPUDirect Async scaffolding
// This header declares the API and device symbols used by the
// host-side libfabric/EFA GDA glue and the device-side posting kernels.
//
// Updated to use official EFA I/O definitions from libfabric fabtests:
// https://github.com/ofiwg/libfabric/tree/main/fabtests/prov/efa/src/efagda

#ifndef NVSHMEM_TRANSPORT_EFA_GDA_H_
#define NVSHMEM_TRANSPORT_EFA_GDA_H_

#include <stdint.h>
#include <stddef.h>

// Libfabric provider extension for EFA:
#include <rdma/fi_ext_efa.h>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

// Include official EFA I/O definitions
#include "../third_party/efa_gda/efa_io_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations to avoid NVSHMEM internals here.
struct fid_domain;
struct fid_ep;

// ----------------------------
// Helper macros for setting EFA descriptor fields
// (from libfabric fabtests cuda_kernel.cu)
// ----------------------------
#define EFA_SET(reg, field, val) \
    do { \
        typeof(reg) tmp_reg = (reg); \
        tmp_reg &= ~(field##_MASK); \
        tmp_reg |= ((val) << (field##_SHIFT)) & (field##_MASK); \
        (reg) = tmp_reg; \
    } while (0)

// ----------------------------
// Device symbols published by host init (defined in .cu)
// ----------------------------
extern __device__ uint8_t*  __nvshmem_efa_sq_buf;     // SQ ring base (device-mapped)
extern __device__ uint32_t* __nvshmem_efa_sq_db;      // SQ doorbell (device-mapped)
extern __device__ uint32_t  __nvshmem_efa_sq_stride;  // bytes per WQE
extern __device__ uint32_t  __nvshmem_efa_sq_size;    // number of WQE slots
extern __device__ uint32_t  __nvshmem_efa_sq_tail;    // producer index (device-side)
extern __device__ uint32_t  __nvshmem_efa_sq_phase;   // phase bit for queue wrap
extern __device__ uint32_t  __nvshmem_efa_sq_mask;    // queue mask (size - 1)

// ----------------------------
// Host-side API
// ----------------------------

/**
 * Initialize EFA GDA for a given libfabric domain + endpoint.
 * - Obtains fi_efa_ops_gda via fi_open_ops(domain, FI_EFA_GDA_OPS, ...)
 * - Queries send/recv WQ attributes (buffer, entry_size, num_entries, doorbell)
 * - Maps/publishes SQ buffer + doorbell to device symbols for GPU access.
 *
 * Returns 0 on success; negative libfabric error codes or CUDA errors otherwise.
 */
int nvshmemt_efa_gda_init_on_ep(struct fid_domain* domain, struct fid_ep* ep);

/**
 * Tear down host-side registrations/mappings created by init.
 * Safe to call multiple times.
 */
void nvshmemt_efa_gda_fini(void);

// ----------------------------
// Device-side API
// ----------------------------

/**
 * Device function to post one RDMA Write ("put") via GDA.
 *
 * @param remote_addr  Remote GPU/host VA to write to.
 * @param src          Local device pointer to payload.
 * @param bytes        Payload size in bytes.
 * @param lkey         Local MR key (for 'src'); pass from host.
 * @param rkey         Remote MR key (for remote_addr).
 * @param dest_qp_num  Destination QP number.
 * @param ah           Address handle for destination.
 *
 * This implementation uses the official EFA WQE structure from efa_io_defs.h.
 */
__device__ void nvshmem_efa_dev_put(void* remote_addr,
                                    const void* src,
                                    size_t bytes,
                                    uint32_t lkey,
                                    uint32_t rkey,
                                    uint16_t dest_qp_num,
                                    uint16_t ah);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NVSHMEM_TRANSPORT_EFA_GDA_H_
