This directory is intended to hold the *authoritative* EFA GDA I/O definitions
that describe the work-queue entry (WQE) layout and opcodes.

Copy (or vendor) the header from libfabric fabtests:

  ofiwg/libfabric/fabtests/prov/efa/src/efagda/efa_io_defs.h

Place it here as:

  third_party/efa_gda/efa_io_defs.h

Then, update include/transport_nvshmem_efa_gda.h to either include it directly,
or define NVSHMEM_EFA_GDA_HAS_REAL_WQE and include the appropriate types.

Until you do this, the code builds with a *placeholder* WQE struct/opcode and
will not produce valid packets.
