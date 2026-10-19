#include "pdma_queue.h"

#include <cerrno>
#include <cstring>
#include <stdexcept>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

namespace hetero_npu {

namespace {
unsigned roundPow2(unsigned v) {
    unsigned p = 1;
    while (p < v) p <<= 1;
    return p;
}
}  // namespace

PdmaDevice::PdmaDevice(const std::string &path, unsigned depth) : path_(path), depth_(depth) {
    if (depth_ == 0 || depth_ > 64) throw std::invalid_argument("queue depth must be 1..64");
    fd_ = open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd_ < 0) throw std::runtime_error("open " + path + ": " + std::strerror(errno));
    int caps = ioctl(fd_, NPU_PDMA_GET_CAPS);
    dmabuf_ = caps >= 0 && (caps & NPU_PDMA_CAPS_DMABUF);
    io_uring_params params{};
    params.flags = IORING_SETUP_SQE128;  // as native/rngd_pdma.c
    int ret = io_uring_queue_init_params(roundPow2(depth_), &ring_, &params);
    if (ret < 0) {
        close(fd_);
        throw std::runtime_error("io_uring_queue_init: " + std::string(std::strerror(-ret)));
    }
}

PdmaDevice::~PdmaDevice() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        pending_.clear();
        while (inflight_ > 0 && !broken_) {  // never leave DMA running past the ring
            fillLocked();
            reapLocked(true);
        }
        while (inflight_ > unsubmitted_.size()) reapLocked(true);
    }
    io_uring_queue_exit(&ring_);
    close(fd_);
}

void PdmaDevice::enqueue(PdmaBatch &batch) {
    std::lock_guard<std::mutex> lock(mu_);
    batch.queued = batch.commands.size();
    batch.inflight = batch.done = 0;
    batch.error = 0;
    batch.bytes = 0;
    for (auto &cmd : batch.commands) {
        cmd.batch = &batch;
        batch.bytes += cmd.bytes;
        pending_.push_back(&cmd);
    }
    fillLocked();
}

void PdmaDevice::progress(bool wait) {
    std::lock_guard<std::mutex> lock(mu_);
    reapLocked(wait);
    fillLocked();
}

void PdmaDevice::drain(PdmaBatch &batch) {
    std::lock_guard<std::mutex> lock(mu_);
    dropQueuedLocked(batch);
    while (batch.inflight > 0) {
        fillLocked();  // retries a deferred submit; other batches keep moving while we wait
        reapLocked(true);
    }
}

void PdmaDevice::fillLocked() {
    if (broken_) return;
    while (inflight_ < depth_ && !pending_.empty()) {
        PdmaCommand *cmd = pending_.front();
        io_uring_sqe *sqe = io_uring_get_sqe(&ring_);
        if (!sqe) break;
        pending_.pop_front();
        io_uring_prep_rw(IORING_OP_URING_CMD, sqe, fd_, nullptr, 0, 0);
        sqe->cmd_op = cmd->op;
        if (cmd->op == NPU_PDMA_URING_CMD_DMABUF_READ || cmd->op == NPU_PDMA_URING_CMD_DMABUF_WRITE) {
            npu_pdma_dmabuf_io io{};
            io.dmabuf_fd = cmd->dmabuf_fd;
            io.desc_count = static_cast<__u32>(cmd->descs.size());
            io.desc_addr = reinterpret_cast<uint64_t>(cmd->descs.data());
            std::memcpy(sqe->cmd, &io, sizeof(io));
        } else {
            npu_pdma_scatter_io io{};
            io.count = cmd->descs.size();
            io.desc_addr = reinterpret_cast<uint64_t>(cmd->descs.data());
            std::memcpy(sqe->cmd, &io, sizeof(io));
        }
        io_uring_sqe_set_data(sqe, cmd);
        cmd->batch->queued--;
        cmd->batch->inflight++;
        inflight_++;
        unsubmitted_.push_back(cmd);
    }
    while (!unsubmitted_.empty()) {
        int ret = io_uring_submit(&ring_);
        if (ret == -EINTR) continue;
        if (ret == -EAGAIN || ret == -EBUSY) return;  // stays in the SQ ring; retried on progress()
        if (ret < 0) {
            // The kernel took none of these SQEs. Fail them and stop using this ring, so they
            // can never be submitted later behind the caller's back.
            broken_ = true;
            while (!unsubmitted_.empty()) {
                PdmaCommand *cmd = unsubmitted_.front();
                unsubmitted_.pop_front();
                finishLocked(*cmd, ret);
            }
            while (!pending_.empty()) {
                PdmaBatch &batch = *pending_.front()->batch;
                dropQueuedLocked(batch);
                if (!batch.active() && batch.on_complete) batch.on_complete(batch);
            }
            return;
        }
        for (int i = 0; i < ret && !unsubmitted_.empty(); i++) unsubmitted_.pop_front();
        if (ret == 0) return;
    }
}

void PdmaDevice::finishLocked(PdmaCommand &cmd, int res) {
    PdmaBatch &batch = *cmd.batch;
    batch.inflight--;
    batch.done++;
    inflight_--;
    if (res < 0 && batch.error == 0) {
        batch.error = res;
        dropQueuedLocked(batch);  // stop issuing the rest of a failed request
    }
    if (!batch.active() && batch.on_complete) batch.on_complete(batch);
}

void PdmaDevice::dropQueuedLocked(PdmaBatch &batch) {
    for (auto it = pending_.begin(); it != pending_.end();) {
        if ((*it)->batch == &batch) {
            it = pending_.erase(it);
            batch.queued--;
            batch.done++;
            if (batch.error == 0) batch.error = -ECANCELED;
        } else {
            ++it;
        }
    }
}

void PdmaDevice::reapLocked(bool wait) {
    bool waited = false;
    while (inflight_ > 0) {
        io_uring_cqe *cqe = nullptr;
        int ret;
        // Never block while SQEs are still waiting to be submitted: nothing may be in the kernel.
        if (wait && !waited && unsubmitted_.empty()) {
            do ret = io_uring_wait_cqe(&ring_, &cqe);
            while (ret == -EINTR);
            waited = true;
        } else {
            ret = io_uring_peek_cqe(&ring_, &cqe);
        }
        if (ret < 0 || !cqe) return;  // -EAGAIN: nothing completed yet
        auto *cmd = static_cast<PdmaCommand *>(io_uring_cqe_get_data(cqe));
        int res = cqe->res;
        io_uring_cqe_seen(&ring_, cqe);
        finishLocked(*cmd, res);
    }
}

}  // namespace hetero_npu
