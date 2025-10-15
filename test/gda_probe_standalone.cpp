// SPDX-License-Identifier: BSD-3-Clause
// Minimal standalone probe for EFA GDA queue exposure.
// Builds and runs *without* NVSHMEM. Useful to verify your libfabric/EFA
// exposes the SQ buffer and doorbell and that CUDA can map them.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_ext_efa.h>

#include <cuda.h>
#include <cuda_runtime.h>

static void die(int rc, const char* where) {
    fprintf(stderr, "ERROR(%d) at %s\n", rc, where);
    std::exit(rc ? rc : 1);
}

int main() {
    int rc = 0;
    struct fi_info *hints = fi_allocinfo();
    if (!hints) die(-1, "fi_allocinfo");

    // Select EFA
    hints->fabric_attr->prov_name = strdup("efa");
    hints->caps = FI_MSG;
    hints->mode = 0;
    hints->domain_attr->mr_mode = FI_MR_VIRT_ADDR | FI_MR_PROV_KEY | FI_MR_ALLOCATED | FI_MR_HMEM;
    hints->ep_attr->type = FI_EP_RDM;

    struct fi_info* info = nullptr;
    rc = fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0, hints, &info);
    if (rc) die(rc, "fi_getinfo(EFA)");

    struct fid_fabric* fabric = nullptr;
    rc = fi_fabric(info->fabric_attr, &fabric, nullptr);
    if (rc) die(rc, "fi_fabric");

    struct fid_domain* domain = nullptr;
    rc = fi_domain(fabric, info, &domain, nullptr);
    if (rc) die(rc, "fi_domain");

    struct fid_ep* ep = nullptr;
    rc = fi_endpoint(domain, info, &ep, nullptr);
    if (rc) die(rc, "fi_endpoint");

    // Create a tiny CQ & counter to satisfy ep enable
    struct fi_cq_attr cq_attr = {};
    cq_attr.format = FI_CQ_FORMAT_MSG;
    cq_attr.size   = 0;
    cq_attr.wait_obj = FI_WAIT_NONE;
    struct fid_cq* cq = nullptr;
    rc = fi_cq_open(domain, &cq_attr, &cq, nullptr);
    if (rc) die(rc, "fi_cq_open");

    struct fi_cntr_attr cntr_attr = {};
    cntr_attr.events = FI_CNTR_EVENTS_COMP;
    struct fid_cntr* cntr = nullptr;
    rc = fi_cntr_open(domain, &cntr_attr, &cntr, nullptr);
    if (rc) die(rc, "fi_cntr_open");

    rc = fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV);
    if (rc) die(rc, "fi_ep_bind(cq)");
    rc = fi_ep_bind(ep, &cntr->fid, FI_WRITE | FI_READ | FI_SEND);
    if (rc) die(rc, "fi_ep_bind(cntr)");

    rc = fi_enable(ep);
    if (rc) die(rc, "fi_enable");

    // Open GDA ops
    const struct fi_efa_ops_gda* gda_ops = nullptr;
    rc = fi_open_ops(&domain->fid, FI_EFA_GDA_OPS, 0, (void**)&gda_ops, NULL);
    if (rc) die(rc, "fi_open_ops(FI_EFA_GDA_OPS)");

    struct fi_efa_wq_attr sq_attr = {};
    struct fi_efa_wq_attr rq_attr = {};
    rc = gda_ops->query_qp_wqs(ep, &sq_attr, &rq_attr);
    if (rc) die(rc, "query_qp_wqs");

    fprintf(stdout, "SQ: buffer=%p entry_size=%zu num_entries=%zu doorbell=%p\n",
            sq_attr.buffer, sq_attr.entry_size, sq_attr.num_entries, sq_attr.doorbell);

    // Try to map buffer/doorbell for device access
    cudaError_t ce;
    void* sq_dev = nullptr;
    ce = cudaHostRegister(sq_attr.buffer,
                          sq_attr.entry_size * sq_attr.num_entries,
                          cudaHostRegisterPortable);
    if (ce != cudaSuccess && ce != cudaErrorHostMemoryAlreadyRegistered) {
        fprintf(stderr, "cudaHostRegister(SQ) failed: %s\n", cudaGetErrorString(ce));
    }
    ce = cudaHostGetDevicePointer(&sq_dev, sq_attr.buffer, 0);
    if (ce != cudaSuccess) die(-1, "cudaHostGetDevicePointer(SQ)");

    fprintf(stdout, "SQ device pointer: %p\n", sq_dev);

    if (sq_attr.doorbell) {
        void* db_dev = nullptr;
        ce = cudaHostRegister(sq_attr.doorbell, sizeof(uint32_t), cudaHostRegisterPortable);
        if (ce != cudaSuccess && ce != cudaErrorHostMemoryAlreadyRegistered) {
            fprintf(stderr, "cudaHostRegister(DB) failed: %s\n", cudaGetErrorString(ce));
        }
        ce = cudaHostGetDevicePointer(&db_dev, sq_attr.doorbell, 0);
        if (ce == cudaSuccess) {
            fprintf(stdout, "DB device pointer: %p\n", db_dev);
        } else {
            fprintf(stderr, "cudaHostGetDevicePointer(DB) failed: %s\n",
                    cudaGetErrorString(ce));
        }
    } else {
        fprintf(stdout, "No device-mappable doorbell was returned.\n");
    }

    // Cleanup
    if (cq) fi_close(&cq->fid);
    if (cntr) fi_close(&cntr->fid);
    if (ep) fi_close(&ep->fid);
    if (domain) fi_close(&domain->fid);
    if (fabric) fi_close(&fabric->fid);
    if (info) fi_freeinfo(info);
    if (hints) fi_freeinfo(hints);

    fprintf(stdout, "GDA probe completed.\n");
    return 0;
}
