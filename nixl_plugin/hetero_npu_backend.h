/* HETERO_NPU: NIXL backend for same-host RNGD -> ATOM KV transfer (docs/nixl-backend.md).
 *
 * Registered memory (nixlBlobDesc.metaInfo selects the kind):
 *   VRAM_SEG "rngd:/dev/rngd/npuNch0"  addr = raw RNGD DRAM offset (furiosa tensor data_ptr)
 *   VRAM_SEG "dmabuf:<fd>[:<offset>]"   exported DMA-BUF (ATOM rblnExportMemoryByDva); addr is a
 *                                       caller-chosen base (e.g. the ATOM DVA) that maps to <offset>
 *   DRAM_SEG ""                          host memory, the staging leg (RNGD <-> host)
 * Transfers are local to the agent (remote agent == local agent). Supported routes:
 *   RNGD -> DMA-BUF  direct P2P, PDMA DMABUF_READ (the validated hetero-npu-p2p direction)
 *   RNGD -> host     PDMA SCATTER_READ  (staged; host copy is explicit, never a fallback)
 *   host -> RNGD     PDMA SCATTER_WRITE (staged)
 * Everything else, including DMA-BUF -> RNGD, is NIXL_ERR_NOT_SUPPORTED.
 *
 * Completion: checkXfer returns NIXL_SUCCESS only after every CQE of the request was reaped
 * with result 0 (the same fence as CopySession.admit). On error it returns NIXL_ERR_BACKEND
 * only once nothing of the request is still in flight; the destination is then undefined.
 */
#ifndef HETERO_NPU_BACKEND_H
#define HETERO_NPU_BACKEND_H

#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "backend/backend_engine.h"
#include "pdma_queue.h"

class nixlHeteroNpuEngine : public nixlBackendEngine {
public:
    static nixl_b_params_t defaultParams();

    explicit nixlHeteroNpuEngine(const nixlBackendInitParams *init_params);
    ~nixlHeteroNpuEngine() override;

    bool supportsRemote() const override { return false; }
    bool supportsLocal() const override { return true; }
    bool supportsNotif() const override { return true; }  // intra-agent, after the fence
    nixl_mem_list_t getSupportedMems() const override { return {DRAM_SEG, VRAM_SEG}; }

    nixl_status_t registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem,
                              nixlBackendMD *&out) override;
    nixl_status_t deregisterMem(nixlBackendMD *meta) override;

    nixl_status_t connect(const std::string &) override { return NIXL_SUCCESS; }
    nixl_status_t disconnect(const std::string &) override { return NIXL_SUCCESS; }
    nixl_status_t unloadMD(nixlBackendMD *) override { return NIXL_SUCCESS; }
    nixl_status_t loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override {
        output = input;
        return NIXL_SUCCESS;
    }

    nixl_status_t prepXfer(const nixl_xfer_op_t &operation, const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote, const std::string &remote_agent,
                           nixlBackendReqH *&handle,
                           const nixl_opt_b_args_t *opt_args = nullptr) const override;
    nixl_status_t postXfer(const nixl_xfer_op_t &operation, const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote, const std::string &remote_agent,
                           nixlBackendReqH *&handle,
                           const nixl_opt_b_args_t *opt_args = nullptr) const override;
    nixl_status_t checkXfer(nixlBackendReqH *handle) const override;
    nixl_status_t releaseReqH(nixlBackendReqH *handle) const override;

    nixl_status_t getNotifs(notif_list_t &notif_list) override;
    nixl_status_t genNotif(const std::string &remote_agent, const std::string &msg) const override;

private:
    std::shared_ptr<hetero_npu::PdmaDevice> openDevice(const std::string &path);
    void pushNotif(const std::string &msg) const;

    unsigned queue_depth_;
    uint64_t desc_bytes_;
    unsigned descs_per_cmd_;
    uint64_t align_;
    bool allow_host_;

    std::mutex devices_mu_;
    std::map<std::string, std::weak_ptr<hetero_npu::PdmaDevice>> devices_;
    mutable std::mutex notif_mu_;
    mutable notif_list_t notifs_;
};

#endif
