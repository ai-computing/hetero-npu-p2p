/* RNGD PDMA command queue: one io_uring per RNGD device file, a bounded number of commands
 * in flight, completion reaped without blocking. No NIXL types here.
 *
 * Same data path as native/rngd_pdma.c (URING_CMD on /dev/rngd/npuNch0); this file adds
 * queue depth > 1 and cancellation of commands that were not yet submitted. A submitted
 * DMA is never aborted: cancel() waits for it.
 */
#ifndef HETERO_NPU_PDMA_QUEUE_H
#define HETERO_NPU_PDMA_QUEUE_H

#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include <liburing.h>
#include <linux/types.h>

#include "npu_pdma_uapi.h"

namespace hetero_npu {

struct PdmaBatch;

// One URING_CMD: up to N scatter descriptors moved by one PDMA request.
struct PdmaCommand {
    uint32_t op = 0;     // NPU_PDMA_URING_CMD_*
    int dmabuf_fd = -1;  // DMABUF_* commands only; one DMA-BUF per command
    std::vector<scatter_io_desc> descs;
    uint64_t bytes = 0;
    PdmaBatch *batch = nullptr;
};

// Commands that complete together (one NIXL request). The batch owns the descriptor memory,
// which the driver may read after submission (io-wq punt), so it must outlive every CQE.
struct PdmaBatch {
    std::vector<PdmaCommand> commands;
    size_t queued = 0;    // waiting for a free queue slot
    size_t inflight = 0;  // submitted, CQE not reaped
    size_t done = 0;      // CQE reaped (or dropped after an error)
    int error = 0;        // first negative CQE result / submission errno
    uint64_t bytes = 0;
    std::function<void(const PdmaBatch &)> on_complete;  // fence reached (success or error)

    bool active() const { return queued + inflight > 0; }
};

class PdmaDevice {
public:
    // Opens the RNGD PDMA device file and an io_uring of `depth` entries. Throws on failure.
    PdmaDevice(const std::string &path, unsigned depth);
    ~PdmaDevice();
    PdmaDevice(const PdmaDevice &) = delete;
    PdmaDevice &operator=(const PdmaDevice &) = delete;

    const std::string &path() const { return path_; }
    bool dmabufCapable() const { return dmabuf_; }

    // Queue every command of the batch (resetting its counters) and submit what fits.
    void enqueue(PdmaBatch &batch);
    // Reap available CQEs and refill. With wait=true, block for at least one CQE if any are
    // in flight on this device.
    void progress(bool wait = false);
    // Drop the batch's unsubmitted commands, then wait for its submitted ones.
    void drain(PdmaBatch &batch);

private:
    void fillLocked();
    void reapLocked(bool wait);
    void finishLocked(PdmaCommand &cmd, int res);
    void dropQueuedLocked(PdmaBatch &batch);

    std::string path_;
    int fd_ = -1;
    bool dmabuf_ = false;
    bool broken_ = false;  // a submit failed hard; the ring is no longer used
    unsigned depth_;
    unsigned inflight_ = 0;
    io_uring ring_{};
    std::deque<PdmaCommand *> pending_;      // waiting for a queue slot
    std::deque<PdmaCommand *> unsubmitted_;  // prepared SQEs the kernel has not taken yet
    std::mutex mu_;
};

}  // namespace hetero_npu

#endif
