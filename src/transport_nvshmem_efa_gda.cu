// SPDX-License-Identifier: BSD-3-Clause
// Device-side implementation for NVSHMEM + EFA GPUDirect Async
// Based on official libfabric fabtests implementation:
// https://github.com/ofiwg/libfabric/blob/main/fabtests/prov/efa/src/efagda/cuda_kernel.cu

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "../include/transport_nvshmem_efa_gda.h"

// Device symbols (storage)
__device__ uint8_t*  __nvshmem_efa_sq_buf   = nullptr;
__device__ uint32_t* __nvshmem_efa_sq_db    = nullptr;
__device__ uint32_t  __nvshmem_efa_sq_stride = 0;
__device__ uint32_t  __nvshmem_efa_sq_size   = 0;
__device__ uint32_t  __nvshmem_efa_sq_tail   = 0;
__device__ uint32_t  __nvshmem_efa_sq_phase  = 1;  // Initial phase is 1
__device__ uint32_t  __nvshmem_efa_sq_mask   = 0;

/**
 * Device function to post RDMA Write via EFA GDA.
 * Implementation follows the official fabtests cuda_kernel.cu pattern.
 */
__device__ void nvshmem_efa_dev_put(void* remote_addr,
                                    const void* src,
                                    size_t bytes,
                                    uint32_t lkey,
                                    uint32_t rkey,
                                    uint16_t dest_qp_num,
                                    uint16_t ah) {
    // 1. Atomically reserve a WQE slot
    uint32_t slot = atomicAdd(&__nvshmem_efa_sq_tail, 1);
    uint32_t local_slot = slot & __nvshmem_efa_sq_mask;

    // 2. Create WQE using official efa_io_tx_wqe structure
    struct efa_io_tx_wqe wqe;
    memset(&wqe, 0, sizeof(wqe));

    // 3. Fill metadata fields
    wqe.meta.dest_qp_num = dest_qp_num;
    wqe.meta.ah = ah;
    wqe.meta.length = bytes;

    // 4. Set operation type to RDMA_WRITE using official macro
    EFA_SET(&wqe.meta.ctrl1, EFA_IO_TX_META_DESC_OP_TYPE, EFA_IO_RDMA_WRITE);

    // 5. Set phase bit (tracks queue wrap-around)
    EFA_SET(&wqe.meta.ctrl2, EFA_IO_TX_META_DESC_PHASE, __nvshmem_efa_sq_phase);

    // 6. Request completion notification
    EFA_SET(&wqe.meta.ctrl2, EFA_IO_TX_META_DESC_COMP_REQ, 1);

    // 7. Mark metadata descriptor as valid
    EFA_SET(&wqe.meta.ctrl1, EFA_IO_TX_META_DESC_META_DESC, 1);

    // 8. Fill RDMA request - remote memory address
    wqe.data.rdma_req.remote_mem_addr.length = bytes;
    wqe.data.rdma_req.remote_mem_addr.rkey = rkey;
    wqe.data.rdma_req.remote_mem_addr.buf_addr = (uint64_t)(uintptr_t)remote_addr;

    // 9. Fill RDMA request - local memory descriptor (scatter-gather list)
    wqe.data.rdma_req.local_mem_desc[0].length = bytes;
    wqe.data.rdma_req.local_mem_desc[0].lkey = lkey;

    // Split 64-bit address into high/low 32-bit fields (EFA hardware requirement)
    uint64_t src_addr = (uint64_t)(uintptr_t)src;
    wqe.data.rdma_req.local_mem_desc[0].buf_addr_lo = src_addr & 0xFFFFFFFF;
    wqe.data.rdma_req.local_mem_desc[0].buf_addr_hi = src_addr >> 32;

    // 10. Calculate WQE offset in queue and copy
    uint32_t sq_desc_offset = local_slot * __nvshmem_efa_sq_stride;
    memcpy(__nvshmem_efa_sq_buf + sq_desc_offset, &wqe, sizeof(wqe));

    // 11. Memory fence to ensure WQE is written before doorbell
    __threadfence_system();

    // 12. Ring doorbell (write producer counter, not slot index)
    if (__nvshmem_efa_sq_db) {
        atomicExch(__nvshmem_efa_sq_db, slot + 1);
    }

    // 13. Update phase when queue wraps around
    // Phase bit helps hardware distinguish new vs old WQEs
    if (local_slot == __nvshmem_efa_sq_mask) {
        atomicAdd(&__nvshmem_efa_sq_phase, 1);
    }
}

/**
 * Test kernel for smoke testing (optional)
 * Launch with single thread to test WQE posting
 */
extern "C" __global__ void __nvshmem_efa_test_put_kernel(void* remote_addr,
                                                         const void* src,
                                                         size_t bytes,
                                                         uint32_t lkey,
                                                         uint32_t rkey,
                                                         uint16_t dest_qp_num,
                                                         uint16_t ah) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        nvshmem_efa_dev_put(remote_addr, src, bytes, lkey, rkey, dest_qp_num, ah);
    }
}
