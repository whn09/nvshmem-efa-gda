// SPDX-License-Identifier: BSD-3-Clause
// NVSHMEM + EFA GPUDirect Async scaffolding
// This header declares the minimal API and device symbols used by the
// host-side libfabric/EFA GDA glue and the device-side posting kernels.
//
// IMPORTANT:
// - This scaffolding *assumes* libfabric's EFA extension header <rdma/fi_ext_efa.h>
//   is available at build time (provided by your EFA-enabled libfabric).
// - The WQE (work queue entry) layout here is a *placeholder*. For a working
//   implementation, replace with the *exact* layout/opcodes from fabtests:
//     ofiwg/libfabric/fabtests/prov/efa/src/efagda/efa_io_defs.h
//   You can place that header into: third_party/efa_gda/efa_io_defs.h
//   and this header will include it automatically.
//
// References (to consult while replacing placeholders):
//   - EFA GDA ops table: <rdma/fi_ext_efa.h>
//   - Host-side usage:   fabtests/prov/efa/src/efa_gda.c
//   - GPU-side sequence: fabtests/prov/efa/src/efagda/cuda_kernel.cu
//
// NOTE:
// To avoid pulling in NVSHMEM internals here, the public init function
// takes raw libfabric handles (fid_domain*, fid_ep*). Your NVSHMEM
// libfabric transport should call nvshmemt_efa_gda_init_on_ep() *after*
// endpoints are enabled and before you start traffic.
//

#ifndef NVSHMEM_TRANSPORT_EFA_GDA_H_
#define NVSHMEM_TRANSPORT_EFA_GDA_H_

#include <stdint.h>
#include <stddef.h>

// Libfabric provider extension for EFA:
#include <rdma/fi_ext_efa.h>

// CUDA headers (device symbol declarations & cudaMemcpyToSymbol from .cpp)
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations to avoid NVSHMEM internals here.
struct fid_domain;
struct fid_ep;

// ----------------------------
// Device symbols published by host init (defined in .cu)
// ----------------------------
extern __device__ uint8_t*  __nvshmem_efa_sq_buf;     // SQ ring base (device-mapped)
extern __device__ uint32_t* __nvshmem_efa_sq_db;      // SQ doorbell (device-mapped), if supported
extern __device__ uint32_t  __nvshmem_efa_sq_stride;  // bytes per WQE
extern __device__ uint32_t  __nvshmem_efa_sq_size;    // number of WQE slots
extern __device__ uint32_t  __nvshmem_efa_sq_tail;    // producer index (device-side)

// ----------------------------
// Placeholder WQE definition
// ----------------------------
// TODO: Replace with *exact* EFA WQE struct and flags/opcodes from
//       fabtests efagda/efa_io_defs.h. You may instead include that file:
//       #include "third_party/efa_gda/efa_io_defs.h"
//
// We intentionally keep a small placeholder to compile the pipeline end-to-end;
// it will not function until replaced with the real layout.
#ifndef NVSHMEM_EFA_GDA_HAS_REAL_WQE
#pragma pack(push, 1)
typedef struct {
    // Minimal fields commonly required for an RDMA Write descriptor.
    // Replace with actual EFA GDA layout:
    uint32_t opcode;          // must match EFA opcode for RDMA write
    uint32_t flags;           // e.g., fence, signaled, etc.
    uint32_t length;          // payload bytes
    uint32_t lkey;            // local MR key for 'src'
    uint64_t src_addr;        // local GPU VA
    uint64_t remote_addr;     // remote VA
    uint32_t rkey;            // remote MR key
    uint32_t reserved;        // align to known size if needed
} EfaWqeRdmaWrite;
#pragma pack(pop)

// Placeholder opcode - must be replaced by the value from efa_io_defs.h
#ifndef EFA_OPCODE_RDMA_WRITE_PLACEHOLDER
#define EFA_OPCODE_RDMA_WRITE_PLACEHOLDER (0x0E) // TODO: replace with real EFA opcode
#endif
#endif // NVSHMEM_EFA_GDA_HAS_REAL_WQE

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
 *
 * NOTE: This placeholder formats a synthetic WQE and rings a doorbell. It will
 * not produce a working packet until you replace the WQE layout/opcode with
 * the EFA GDA definitions from efa_io_defs.h (see TODO above).
 */
__device__ void nvshmem_efa_dev_put(void* remote_addr,
                                    const void* src,
                                    size_t bytes,
                                    uint32_t lkey,
                                    uint32_t rkey);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NVSHMEM_TRANSPORT_EFA_GDA_H_
