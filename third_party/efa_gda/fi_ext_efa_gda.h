/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * EFA GDA (GPU Direct Async) extension definitions
 * This file provides the GDA-specific extensions for libfabric EFA provider
 */

#ifndef _FI_EXT_EFA_GDA_H_
#define _FI_EXT_EFA_GDA_H_

#include <rdma/fi_endpoint.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* GDA ops identifier for fi_open_ops */
#define FI_EFA_GDA_OPS "efa gda ops"

/**
 * struct fi_efa_wq_attr - Work Queue attributes for GDA
 * @buffer: Pointer to the work queue buffer (SQ or RQ)
 * @entry_size: Size of each work queue entry in bytes
 * @num_entries: Number of entries in the work queue
 * @doorbell: Pointer to the doorbell register (may be NULL if not device-mappable)
 */
struct fi_efa_wq_attr {
    void *buffer;
    size_t entry_size;
    size_t num_entries;
    volatile uint32_t *doorbell;
};

/**
 * struct fi_efa_ops_gda - GDA operations interface
 * @query_qp_wqs: Query the Send Queue and Receive Queue work queue attributes
 *                for a given endpoint
 *
 * @ep: The libfabric endpoint (fid_ep)
 * @sq_attr: Output parameter for Send Queue attributes
 * @rq_attr: Output parameter for Receive Queue attributes
 * @return: 0 on success, negative error code on failure
 */
struct fi_efa_ops_gda {
    int (*query_qp_wqs)(struct fid_ep *ep,
                        struct fi_efa_wq_attr *sq_attr,
                        struct fi_efa_wq_attr *rq_attr);
};

#ifdef __cplusplus
}
#endif

#endif /* _FI_EXT_EFA_GDA_H_ */
