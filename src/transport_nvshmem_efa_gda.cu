// SPDX-License-Identifier: BSD-3-Clause
#include <stdint.h>
#include <stddef.h>
#include "../include/transport_nvshmem_efa_gda.h"

// Device symbols (storage)
__device__ uint8_t*  __nvshmem_efa_sq_buf   = nullptr;
__device__ uint32_t* __nvshmem_efa_sq_db    = nullptr;
__device__ uint32_t  __nvshmem_efa_sq_stride = 0;
__device__ uint32_t  __nvshmem_efa_sq_size   = 0;
__device__ uint32_t  __nvshmem_efa_sq_tail   = 0;

// Helpers for pointer arithmetic
static __device__ __forceinline__ uint8_t* sq_slot_ptr(uint32_t slot) {
    uint8_t* base = __nvshmem_efa_sq_buf;
    uint32_t stride = __nvshmem_efa_sq_stride;
    return base + (size_t)slot * (size_t)stride;
}

// NOTE: This posting path is a *placeholder* until you replace the WQE struct
// and opcode with the actual ones from efagda/efa_io_defs.h. The sequence and
// fences mirror GPUDirect Async best practices (write WQE, fence, ring DB).
__device__ void nvshmem_efa_dev_put(void* remote_addr,
                                    const void* src,
                                    size_t bytes,
                                    uint32_t lkey,
                                    uint32_t rkey) {
    // Reserve a WQE slot (ring buffer). For now a simple atomic tail.
    uint32_t slot = atomicAdd(&__nvshmem_efa_sq_tail, 1);
    slot %= (__nvshmem_efa_sq_size ? __nvshmem_efa_sq_size : 1u);

    // Format placeholder WQE
    EfaWqeRdmaWrite wqe;
#ifdef EFA_OPCODE_RDMA_WRITE_PLACEHOLDER
    wqe.opcode = (uint32_t)EFA_OPCODE_RDMA_WRITE_PLACEHOLDER;
#else
    // If the real opcode macro is available from efa_io_defs.h, use it here.
    wqe.opcode = 0; // TODO: replace
#endif
    wqe.flags  = 0; // TODO: signaled/fence bits as required by EFA
    wqe.length = (uint32_t)bytes;
    wqe.lkey   = lkey;
    wqe.src_addr    = (uint64_t)(uintptr_t)src;
    wqe.remote_addr = (uint64_t)(uintptr_t)remote_addr;
    wqe.rkey        = rkey;
    wqe.reserved    = 0;

    // Write WQE into the SQ at the computed slot
    uint8_t* dst = sq_slot_ptr(slot);
    // Assumes WQE fits in entry_size; the real layout must match entry_size.
    // A plain memcpy is sufficient because WQE is POD.
    uint8_t* src_wqe = reinterpret_cast<uint8_t*>(&wqe);
    for (size_t i = 0; i < sizeof(EfaWqeRdmaWrite); ++i) {
        dst[i] = src_wqe[i];
    }

    // Make writes visible to NIC before ringing doorbell
    __threadfence_system();

    // Ring doorbell if mapped for device (some stacks require writing producer index)
    if (__nvshmem_efa_sq_db) {
        // Many providers require writing "new tail" or "+= 1" semantics.
        // Here we write (slot+1) as a plausible producer index; adapt to EFA needs.
        *__nvshmem_efa_sq_db = slot + 1;
        __threadfence_system();
    }
}

// A tiny kernel you can use for smoke tests (optional)
extern "C" __global__ void __nvshmem_efa_test_put_kernel(void* remote_addr,
                                                         const void* src,
                                                         size_t bytes,
                                                         uint32_t lkey,
                                                         uint32_t rkey) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        nvshmem_efa_dev_put(remote_addr, src, bytes, lkey, rkey);
    }
}
