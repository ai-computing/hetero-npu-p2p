#include "hetero_npu_backend.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <regex>
#include <stdexcept>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "common/nixl_log.h"
#include "npu_address_map.h"  // NPU_DRAM_INWARD_ADDR, as native/rngd_pdma.c

using hetero_npu::PdmaBatch;
using hetero_npu::PdmaCommand;
using hetero_npu::PdmaDevice;

namespace {

// Same RNGD DRAM window the native helper validates (48 GiB minus the firmware reserve).
constexpr uint64_t kRngdDramLimit = 48ULL * 1024 * 1024 * 1024 - 512ULL * 1024 * 1024;
// Keep each command well under the driver's 4 GiB-per-DMA-BUF-command bound.
constexpr uint64_t kMaxCommandBytes = 1ULL << 30;

enum class Kind { Rngd, Dmabuf, Host };

struct Region : public nixlBackendMD {
    Kind kind;
    uintptr_t base;
    size_t len;
    std::shared_ptr<PdmaDevice> device;  // Rngd
    int dmabuf_fd = -1;                  // Dmabuf: our dup, closed on deregistration
    uint64_t dmabuf_offset = 0;

    Region(Kind k, uintptr_t b, size_t l) : nixlBackendMD(true), kind(k), base(b), len(l) {}
    ~Region() override {
        if (dmabuf_fd >= 0) close(dmabuf_fd);
    }
};

struct Request : public nixlBackendReqH {
    std::shared_ptr<PdmaDevice> device;
    PdmaBatch batch;
    bool posted = false;
};

bool parseUnsigned(const std::string &s, uint64_t &v) {
    if (s.empty() || s[0] == '-') return false;
    char *end = nullptr;
    errno = 0;
    v = std::strtoull(s.c_str(), &end, 0);
    return errno == 0 && end && *end == '\0';
}

uint64_t param(const nixl_b_params_t &params, const char *key, uint64_t fallback) {
    auto it = params.find(key);
    uint64_t v;
    if (it == params.end() || !parseUnsigned(it->second, v)) return fallback;
    return v;
}

nixl_status_t errnoStatus(int err) {
    return err == -ECANCELED ? NIXL_ERR_CANCELED : NIXL_ERR_BACKEND;
}

}  // namespace

nixl_b_params_t nixlHeteroNpuEngine::defaultParams() {
    // queue_depth 1 reproduces the sequential native helper; desc_bytes/descs_per_cmd match its
    // validated 2 MiB x 8 grouping. allow_host=0 rejects DRAM_SEG routes.
    return {{"queue_depth", "1"}, {"desc_bytes", "2097152"}, {"descs_per_cmd", "8"},
            {"align", "4096"}, {"allow_host", "1"}};
}

nixlHeteroNpuEngine::nixlHeteroNpuEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params) {
    const auto &p = getCustomParams();
    queue_depth_ = static_cast<unsigned>(param(p, "queue_depth", 1));
    desc_bytes_ = param(p, "desc_bytes", 2 << 20);
    descs_per_cmd_ = static_cast<unsigned>(param(p, "descs_per_cmd", 8));
    align_ = param(p, "align", 4096);
    allow_host_ = param(p, "allow_host", 1) != 0;
    if (queue_depth_ < 1 || queue_depth_ > 64 || desc_bytes_ == 0 || descs_per_cmd_ < 1 ||
        align_ == 0 || desc_bytes_ % align_ || desc_bytes_ > kMaxCommandBytes) {
        NIXL_ERROR << "HETERO_NPU: invalid parameters";
        initErr = true;
    }
}

nixlHeteroNpuEngine::~nixlHeteroNpuEngine() = default;

std::shared_ptr<PdmaDevice> nixlHeteroNpuEngine::openDevice(const std::string &path) {
    std::lock_guard<std::mutex> lock(devices_mu_);
    if (auto dev = devices_[path].lock()) return dev;
    auto dev = std::make_shared<PdmaDevice>(path, queue_depth_);
    devices_[path] = dev;
    return dev;
}

nixl_status_t nixlHeteroNpuEngine::registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem,
                                               nixlBackendMD *&out) {
    const std::string &meta = mem.metaInfo;
    if (mem.len == 0) return NIXL_ERR_INVALID_PARAM;
    try {
        if (nixl_mem == DRAM_SEG) {
            if (!allow_host_ || mem.addr == 0) return NIXL_ERR_NOT_ALLOWED;
            out = new Region(Kind::Host, mem.addr, mem.len);
            return NIXL_SUCCESS;
        }
        if (nixl_mem != VRAM_SEG) return NIXL_ERR_NOT_SUPPORTED;

        static const std::regex rngd_re("rngd:(/dev/rngd/npu(0|[1-9][0-9]*)ch0)");
        static const std::regex dmabuf_re("dmabuf:([0-9]+)(:([0-9]+|0x[0-9a-fA-F]+))?");
        std::smatch m;
        if (std::regex_match(meta, m, rngd_re)) {
            // A raw DRAM offset of a live RNGD allocation; the owner keeps it alive.
            if (mem.addr == 0 || mem.addr % 64 || mem.len > kRngdDramLimit ||
                mem.addr > kRngdDramLimit - mem.len)
                return NIXL_ERR_INVALID_PARAM;
            auto *r = new Region(Kind::Rngd, mem.addr, mem.len);
            r->device = openDevice(m[1].str());
            out = r;
            return NIXL_SUCCESS;
        }
        if (std::regex_match(meta, m, dmabuf_re)) {
            uint64_t fd, offset = 0;
            if (!parseUnsigned(m[1].str(), fd) || (m[3].matched && !parseUnsigned(m[3].str(), offset)))
                return NIXL_ERR_INVALID_PARAM;
            struct stat st {};
            if (fstat(static_cast<int>(fd), &st) || st.st_size < 0 ||
                offset > static_cast<uint64_t>(st.st_size) ||
                mem.len > static_cast<uint64_t>(st.st_size) - offset)
                return NIXL_ERR_INVALID_PARAM;
            // Our own reference: the DMA-BUF stays valid even if the caller closes its fd. The
            // exporting SDK allocation must still outlive every transfer (caller's contract).
            int dup_fd = fcntl(static_cast<int>(fd), F_DUPFD_CLOEXEC, 0);
            if (dup_fd < 0) return NIXL_ERR_BACKEND;
            auto *r = new Region(Kind::Dmabuf, mem.addr, mem.len);
            r->dmabuf_fd = dup_fd;
            r->dmabuf_offset = offset;
            out = r;
            return NIXL_SUCCESS;
        }
        NIXL_ERROR << "HETERO_NPU: VRAM metaInfo must be 'rngd:/dev/rngd/npuNch0' or 'dmabuf:<fd>', got '"
                   << meta << "'";
        return NIXL_ERR_INVALID_PARAM;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "HETERO_NPU: registerMem: " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t nixlHeteroNpuEngine::deregisterMem(nixlBackendMD *meta) {
    delete static_cast<Region *>(meta);
    return NIXL_SUCCESS;
}

nixl_status_t nixlHeteroNpuEngine::prepXfer(const nixl_xfer_op_t &operation,
                                            const nixl_meta_dlist_t &local,
                                            const nixl_meta_dlist_t &remote,
                                            const std::string &remote_agent,
                                            nixlBackendReqH *&handle,
                                            const nixl_opt_b_args_t *) const {
    if (remote_agent != localAgent) return NIXL_ERR_NOT_SUPPORTED;  // same host, same agent
    const int count = local.descCount();
    if (count == 0 || count != remote.descCount()) return NIXL_ERR_INVALID_PARAM;
    // NIXL_WRITE moves local -> remote, NIXL_READ moves remote -> local.
    const auto &src = operation == NIXL_WRITE ? local : remote;
    const auto &dst = operation == NIXL_WRITE ? remote : local;

    auto req = std::make_unique<Request>();
    uint32_t op = 0;
    PdmaCommand *cmd = nullptr;
    for (int i = 0; i < count; i++) {
        const nixlMetaDesc &s = src[i];
        const nixlMetaDesc &d = dst[i];
        auto *sr = static_cast<Region *>(s.metadataP);
        auto *dr = static_cast<Region *>(d.metadataP);
        if (!sr || !dr || s.len != d.len || s.len == 0) return NIXL_ERR_INVALID_PARAM;
        if (s.addr < sr->base || s.len > sr->len || s.addr - sr->base > sr->len - s.len ||
            d.addr < dr->base || d.len > dr->len || d.addr - dr->base > dr->len - d.len)
            return NIXL_ERR_INVALID_PARAM;
        if (s.addr % align_ || d.addr % align_ || s.len % align_) return NIXL_ERR_INVALID_PARAM;

        uint32_t this_op;
        std::shared_ptr<PdmaDevice> dev;
        if (sr->kind == Kind::Rngd && dr->kind == Kind::Dmabuf) {
            this_op = NPU_PDMA_URING_CMD_DMABUF_READ;
            dev = sr->device;
            if (!dev->dmabufCapable()) return NIXL_ERR_NOT_SUPPORTED;
        } else if (sr->kind == Kind::Rngd && dr->kind == Kind::Host) {
            this_op = NPU_PDMA_URING_CMD_SCATTER_READ;
            dev = sr->device;
        } else if (sr->kind == Kind::Host && dr->kind == Kind::Rngd) {
            this_op = NPU_PDMA_URING_CMD_SCATTER_WRITE;
            dev = dr->device;
        } else {
            // DMA-BUF -> RNGD and RNGD -> RNGD are not validated; host <-> DMA-BUF is not PDMA.
            return NIXL_ERR_NOT_SUPPORTED;
        }
        if (i == 0) {
            op = this_op;
            req->device = dev;
        } else if (this_op != op || dev != req->device) {
            return NIXL_ERR_NOT_SUPPORTED;  // one route and one RNGD device per request
        }

        const int fd = this_op == NPU_PDMA_URING_CMD_DMABUF_READ ? dr->dmabuf_fd : -1;
        for (uint64_t off = 0; off < s.len; off += desc_bytes_) {
            const uint64_t len = std::min<uint64_t>(desc_bytes_, s.len - off);
            scatter_io_desc desc{};
            desc.len = len;
            switch (this_op) {
            case NPU_PDMA_URING_CMD_DMABUF_READ:
                desc.src = NPU_DRAM_INWARD_ADDR + s.addr + off;
                desc.dst = dr->dmabuf_offset + (d.addr - dr->base) + off;
                break;
            case NPU_PDMA_URING_CMD_SCATTER_READ:
                desc.src = NPU_DRAM_INWARD_ADDR + s.addr + off;
                desc.dst = d.addr + off;
                break;
            default:  // SCATTER_WRITE
                desc.src = s.addr + off;
                desc.dst = NPU_DRAM_INWARD_ADDR + d.addr + off;
            }
            if (!cmd || cmd->dmabuf_fd != fd || cmd->descs.size() >= descs_per_cmd_ ||
                cmd->bytes + len > kMaxCommandBytes) {
                req->batch.commands.emplace_back();
                cmd = &req->batch.commands.back();
                cmd->op = this_op;
                cmd->dmabuf_fd = fd;
            }
            cmd->descs.push_back(desc);
            cmd->bytes += len;
        }
    }
    // commands is not resized after this point, so PdmaCommand addresses stay stable.
    handle = req.release();
    return NIXL_SUCCESS;
}

nixl_status_t nixlHeteroNpuEngine::postXfer(const nixl_xfer_op_t &, const nixl_meta_dlist_t &,
                                            const nixl_meta_dlist_t &, const std::string &,
                                            nixlBackendReqH *&handle,
                                            const nixl_opt_b_args_t *opt_args) const {
    auto *req = static_cast<Request *>(handle);
    if (req->batch.active()) return NIXL_ERR_REPOST_ACTIVE;
    const bool notify = opt_args && opt_args->hasNotif;
    const std::string msg = notify ? opt_args->notifMsg : std::string();
    // The notification is generated only on a clean fence, never for a failed transfer.
    req->batch.on_complete = [this, notify, msg](const PdmaBatch &b) {
        if (notify && b.error == 0) pushNotif(msg);
    };
    try {
        req->device->enqueue(req->batch);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "HETERO_NPU: postXfer: " << e.what();
        return NIXL_ERR_BACKEND;
    }
    req->posted = true;
    if (req->batch.active()) return NIXL_IN_PROG;
    return req->batch.error ? errnoStatus(req->batch.error) : NIXL_SUCCESS;
}

nixl_status_t nixlHeteroNpuEngine::checkXfer(nixlBackendReqH *handle) const {
    auto *req = static_cast<Request *>(handle);
    if (!req->posted) return NIXL_ERR_NOT_POSTED;
    if (req->batch.active()) req->device->progress(false);
    if (req->batch.active()) return NIXL_IN_PROG;
    if (req->batch.error) {
        NIXL_ERROR << "HETERO_NPU: transfer failed: " << std::strerror(-req->batch.error);
        return errnoStatus(req->batch.error);
    }
    return NIXL_SUCCESS;
}

nixl_status_t nixlHeteroNpuEngine::releaseReqH(nixlBackendReqH *handle) const {
    auto *req = static_cast<Request *>(handle);
    // A submitted PDMA cannot be aborted: drop what was not submitted and wait for the rest,
    // so the descriptor memory and the destination are never released under a running DMA.
    if (req->posted && req->batch.active()) req->device->drain(req->batch);
    delete req;
    return NIXL_SUCCESS;
}

void nixlHeteroNpuEngine::pushNotif(const std::string &msg) const {
    std::lock_guard<std::mutex> lock(notif_mu_);
    notifs_.emplace_back(localAgent, msg);
}

nixl_status_t nixlHeteroNpuEngine::getNotifs(notif_list_t &notif_list) {
    std::vector<std::shared_ptr<PdmaDevice>> live;
    {
        std::lock_guard<std::mutex> lock(devices_mu_);
        for (auto &kv : devices_)
            if (auto dev = kv.second.lock()) live.push_back(dev);
    }
    for (auto &dev : live) dev->progress(false);  // completions become notifications here too
    std::lock_guard<std::mutex> lock(notif_mu_);
    notif_list.insert(notif_list.end(), notifs_.begin(), notifs_.end());
    notifs_.clear();
    return NIXL_SUCCESS;
}

nixl_status_t nixlHeteroNpuEngine::genNotif(const std::string &remote_agent,
                                            const std::string &msg) const {
    if (remote_agent != localAgent) return NIXL_ERR_NOT_SUPPORTED;
    pushNotif(msg);
    return NIXL_SUCCESS;
}
