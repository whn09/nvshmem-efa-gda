# NVSHMEM EFA GDA Scaffolding (Drop-in)

This bundle gives you **buildable scaffolding** to wire Amazon EFA *GPU Direct Async (GDA)* into
NVSHMEM’s libfabric transport and to **publish** the EFA Send Queue (SQ) and **doorbell** to the GPU.
It also includes a **standalone probe** you can run inside your container to verify queue exposure.

> ⚠️ **Important**: The device posting path uses a **placeholder WQE** and **placeholder opcode**.
> To actually transmit packets, replace them with the real EFA GDA definitions from fabtests:
>
>   `ofiwg/libfabric/fabtests/prov/efa/src/efagda/efa_io_defs.h`
>
> Place that file at: `third_party/efa_gda/efa_io_defs.h` and update the header include accordingly.

---

## Tree

```
include/
  transport_nvshmem_efa_gda.h
src/
  transport_nvshmem_efa_gda.cpp
  transport_nvshmem_efa_gda.cu
test/
  gda_probe_standalone.cpp
scripts/
  build_standalone_probe.sh
  install_into_nvshmem.sh
  run_all_in_container.sh
third_party/efa_gda/
  README.md
```

## What it does

- **Host init (C++)**: calls `fi_open_ops(..., FI_EFA_GDA_OPS, ...)`, queries SQ/RQ attributes,
  maps the SQ buffer + doorbell for device access, and **publishes pointers** to device symbols.
- **Device post (CUDA)**: formats a WQE and rings the doorbell (placeholders), with proper
  `__threadfence_system()` ordering. Swap in the *real* layout/opcodes for EFA GDA.
- **Standalone probe**: independent test that verifies libfabric exposes a mappable SQ buffer and doorbell.

## Quick start in your Docker

1) Copy `nvshmem-efa-gda.zip` into the container.
2) Inside the container:
   ```bash
   /root/nvshmem-efa-gda/scripts/run_all_in_container.sh /root/nvshmem-efa-gda.zip
   ```
   This script:
   - Unzips into `/root/nvshmem-efa-gda/`
   - Builds and runs `/workspace/gda_probe`
   - Downloads NVSHMEM source and injects the new files with sed-based patches
   - Rebuilds NVSHMEM (if a build dir exists; otherwise prints how to build)
   - Prints environment exports

## Manual integration (if sed fails)

- Add this include to the libfabric transport TU (near other headers):
  ```c++
  #include "transports/libfabric/gda/transport_nvshmem_efa_gda.h"
  ```
- After endpoints are enabled in `nvshmemt_libfabric_connect_endpoints`, add:
  ```c++
  nvshmemt_efa_gda_init_on_ep(state->domain, state->eps[0].endpoint);
  ```
  (Use the appropriate EP index for your send path.)
- In `nvshmemt_libfabric_finalize` before tearing down domain/fabric:
  ```c++
  nvshmemt_efa_gda_fini();
  ```
- In `transports/libfabric/CMakeLists.txt`, add:
  ```cmake
  add_library(nvshmem_efa_gda_objs OBJECT
    gda/transport_nvshmem_efa_gda.cpp
    gda/transport_nvshmem_efa_gda.cu
  )
  set_property(TARGET nvshmem_efa_gda_objs PROPERTY CUDA_STANDARD 17)
  target_sources(nvshmem_transport_libfabric PRIVATE $<TARGET_OBJECTS:nvshmem_efa_gda_objs>)
  ```

## Replace the placeholder WQE

- Copy from fabtests: `efa_io_defs.h` → `third_party/efa_gda/`.
- In `include/transport_nvshmem_efa_gda.h`, either include it directly or
  set a compile definition to use it.
- Update `src/transport_nvshmem_efa_gda.cu`:
  - Replace `EfaWqeRdmaWrite` with the real EFA WQE type(s)
  - Replace `EFA_OPCODE_RDMA_WRITE_PLACEHOLDER` with the real opcode
  - Fill **all required fields** (producer index, sequence/tokens, flags)
  - Match doorbell semantics from fabtests `cuda_kernel.cu` (write “new tail” or WQE count)

## Smoke test pattern

After init succeeds and device symbols are published, you can launch (one-block) test:
```c++
extern "C" __global__ void __nvshmem_efa_test_put_kernel(void* raddr,
                                                         const void* src,
                                                         size_t n,
                                                         uint32_t lkey,
                                                         uint32_t rkey);
```

## Environment

You’ll typically need:
```
export NVSHMEM_REMOTE_TRANSPORT=libfabric
export NVSHMEM_LIBFABRIC_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
```

## Notes

- The scaffolding uses `cudaHostRegister` + `cudaHostGetDevicePointer` to map SQ/DB.
  Many libfabric builds already pin/map these buffers such that mapping succeeds.
- If your provider does not expose a device-mappable doorbell, you can use a
  **doorbell record** in memory and have the **host** ring the MMIO doorbell, as shown
  in fabtests. That fallback is not implemented here but can be added easily.
- This bundle does not ship third-party headers; you must vendor the EFA GDA I/O
  definitions yourself for a working datapath.
