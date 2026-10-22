/* Completion-fenced RNGD PDMA <-> imported DMA-BUF (e.g. CUDA memory exported by the NVIDIA open kernel module).
 *
 * Shared library loaded in-process by hetero_p2p.gpu.RngdDmabufBackend (ctypes). One context = one RNGD PDMA
 * device file and one io_uring that stay open across transfers, so a transfer costs no process start, no device
 * open and no ring setup.
 *
 *   rd_open(path, queue_depth, &ctx)   open /dev/rngd/npuNch0, require NPU_PDMA_CAPS_DMABUF, set up the ring
 *   rd_open2(path, queue_depth, flags, &ctx)   flags RD_POLL: completion-polled ring (IORING_SETUP_IOPOLL); the
 *                                      driver then reaps the PDMA status from io_uring_enter instead of a workqueue
 *   rd_profile(ctx, ns[3])             last rd_run: validated, first submission returned, fenced (ns from entry)
 *   rd_run(ctx, write, ...)            run commands with up to queue_depth in flight; return after every CQE
 *   rd_close(ctx)
 *
 * write = 0: RNGD DRAM -> DMA-BUF (NPU_PDMA_URING_CMD_DMABUF_READ); write = 1: DMA-BUF -> RNGD DRAM (..._WRITE).
 * Command c moves counts[c] consecutive descriptors through the DMA-BUF fds[c]. Descriptor i copies len[i] bytes
 * between RNGD DRAM offset npu[i] and DMA-BUF offset off[i]; all 4 KiB aligned.
 *
 * The whole request is validated before the first submission (-EINVAL, nothing submitted). After a failed
 * command no further command is submitted, but every submitted one is still reaped: rd_run never returns with
 * DMA in flight. If the ring itself fails, the context refuses further work (-EIO).
 *
 * io_uring is used through raw syscalls (no liburing), with 128-byte SQEs as the driver's uring_cmd requires.
 * No host data buffers and no host-copy fallback.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <linux/io_uring.h>
#include "npu_pdma_uapi.h"
#include "npu_address_map.h"

#define RD_ABI 2
#define RD_POLL 1U
#define RD_SETUP_IOPOLL (1U << 0)        /* IORING_SETUP_IOPOLL */
#define RD_OP_URING_CMD 46               /* IORING_OP_URING_CMD (Linux >= 5.19) */
#define RD_SETUP_SQE128 (1U << 10)       /* IORING_SETUP_SQE128 */
#define RD_SQE_BYTES 128
#define RD_PAGE 4096ULL
#define RD_MAX_DEPTH 64U
#define RD_MAX_CMD_BYTES 0xFFFFFFFFULL   /* the driver sums one command's descriptors into a u32 */
/* RNGD DRAM usable for PDMA: NPU_DRAM_SIZE - NPU_FW_SIZE (npu_pdma_uring_cmd.c), as rngd_pdma.c */
#define RD_DRAM_LIMIT (48ULL * 1024 * 1024 * 1024 - 512ULL * 1024 * 1024)

struct rd_cqe { uint64_t user_data; int32_t res; uint32_t flags; };

struct rd_ctx {
  int dev, ring, broken;
  unsigned depth;
  void *sq_map, *cq_map; size_t sq_len, cq_len;
  uint8_t *sqes; size_t sqes_len;
  unsigned *sq_head, *sq_tail, *sq_mask, *sq_array, *cq_head, *cq_tail, *cq_mask;
  struct rd_cqe *cqes;
  uint64_t profile[3];
};

static uint64_t now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ULL + (uint64_t)t.tv_nsec; }

void rd_profile(const struct rd_ctx *c, uint64_t out[3]) { if (c && out) memcpy(out, c->profile, sizeof c->profile); }

int rd_abi(void) { return RD_ABI; }

static int enter(int fd, unsigned submit, unsigned wait) {
  long r = syscall(__NR_io_uring_enter, fd, submit, wait, wait ? IORING_ENTER_GETEVENTS : 0, NULL, 0);
  return r < 0 ? -errno : (int)r;
}

void rd_close(struct rd_ctx *c) {
  if (!c) return;
  if (c->sqes && c->sqes != MAP_FAILED) munmap(c->sqes, c->sqes_len);
  if (c->cq_map && c->cq_map != MAP_FAILED && c->cq_map != c->sq_map) munmap(c->cq_map, c->cq_len);
  if (c->sq_map && c->sq_map != MAP_FAILED) munmap(c->sq_map, c->sq_len);
  if (c->ring >= 0) close(c->ring);
  if (c->dev >= 0) close(c->dev);
  free(c);
}

int rd_open2(const char *path, unsigned depth, unsigned flags, struct rd_ctx **out) {
  unsigned id; char extra; char expected[64];
  if (!path || !out || !depth || depth > RD_MAX_DEPTH || (flags & ~RD_POLL)) return -EINVAL;
  if (sscanf(path, "/dev/rngd/npu%uch0%c", &id, &extra) != 1) return -EINVAL;
  snprintf(expected, sizeof expected, "/dev/rngd/npu%uch0", id);
  if (strcmp(expected, path)) return -EINVAL;
  struct rd_ctx *c = calloc(1, sizeof *c);
  if (!c) return -ENOMEM;
  c->dev = c->ring = -1; c->depth = depth;
  int err;
  if ((c->dev = open(path, O_RDWR | O_CLOEXEC)) < 0) { err = -errno; goto fail; }
  int caps = ioctl(c->dev, NPU_PDMA_GET_CAPS);
  if (caps < 0 || !(caps & NPU_PDMA_CAPS_DMABUF)) { err = -EOPNOTSUPP; goto fail; }
  struct io_uring_params p; memset(&p, 0, sizeof p); p.flags = RD_SETUP_SQE128 | ((flags & RD_POLL) ? RD_SETUP_IOPOLL : 0);
  long r = syscall(__NR_io_uring_setup, depth, &p);
  if (r < 0) { err = -errno; goto fail; }
  c->ring = (int)r;
  c->sq_len = p.sq_off.array + p.sq_entries * sizeof(unsigned);
  c->cq_len = p.cq_off.cqes + p.cq_entries * sizeof(struct rd_cqe);
  if (p.features & IORING_FEAT_SINGLE_MMAP) { if (c->cq_len > c->sq_len) c->sq_len = c->cq_len; c->cq_len = c->sq_len; }
  c->sq_map = mmap(NULL, c->sq_len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, c->ring, IORING_OFF_SQ_RING);
  if (c->sq_map == MAP_FAILED) { err = -errno; goto fail; }
  if (p.features & IORING_FEAT_SINGLE_MMAP) c->cq_map = c->sq_map;
  else {
    c->cq_map = mmap(NULL, c->cq_len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, c->ring, IORING_OFF_CQ_RING);
    if (c->cq_map == MAP_FAILED) { err = -errno; goto fail; }
  }
  c->sqes_len = (size_t)p.sq_entries * RD_SQE_BYTES;
  c->sqes = mmap(NULL, c->sqes_len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, c->ring, IORING_OFF_SQES);
  if (c->sqes == MAP_FAILED) { err = -errno; goto fail; }
  uint8_t *sq = c->sq_map, *cq = c->cq_map;
  c->sq_head = (unsigned *)(sq + p.sq_off.head); c->sq_tail = (unsigned *)(sq + p.sq_off.tail);
  c->sq_mask = (unsigned *)(sq + p.sq_off.ring_mask); c->sq_array = (unsigned *)(sq + p.sq_off.array);
  c->cq_head = (unsigned *)(cq + p.cq_off.head); c->cq_tail = (unsigned *)(cq + p.cq_off.tail);
  c->cq_mask = (unsigned *)(cq + p.cq_off.ring_mask); c->cqes = (struct rd_cqe *)(cq + p.cq_off.cqes);
  if (p.sq_entries < depth || p.cq_entries < depth) { err = -EINVAL; goto fail; }
  *out = c;
  return 0;
fail:
  rd_close(c);
  return err;
}

int rd_open(const char *path, unsigned depth, struct rd_ctx **out) { return rd_open2(path, depth, 0, out); }

/* Reap every available CQE; returns how many. */
static unsigned reap(struct rd_ctx *c, int *first_error, uint64_t *completed) {
  unsigned head = *c->cq_head, n = 0;
  while (head != __atomic_load_n(c->cq_tail, __ATOMIC_ACQUIRE)) {
    struct rd_cqe *e = &c->cqes[head & *c->cq_mask];
    if (e->res == 0) (*completed)++;
    else if (!*first_error) *first_error = e->res < 0 ? e->res : -EIO;
    head++; n++;
  }
  __atomic_store_n(c->cq_head, head, __ATOMIC_RELEASE);
  return n;
}

int rd_run(struct rd_ctx *c, int write, uint32_t ncmd, const int32_t *fds, const uint32_t *counts,
           uint64_t ndesc, const uint64_t *npu, const uint64_t *off, const uint64_t *len, uint64_t *completed) {
  if (!c || !completed) return -EINVAL;
  uint64_t t0 = now_ns();
  memset(c->profile, 0, sizeof c->profile);
  *completed = 0;
  if (c->broken) return -EIO;
  if (!ncmd || !ndesc || !fds || !counts || !npu || !off || !len) return -EINVAL;
  /* Validate everything before the first submission. */
  uint64_t first = 0;
  for (uint32_t k = 0; k < ncmd; k++) {
    struct stat st; uint64_t bytes = 0;
    if (!counts[k] || counts[k] > ndesc - first || fstat(fds[k], &st) || st.st_size <= 0) return -EINVAL;
    for (uint64_t i = first; i < first + counts[k]; i++) {
      if (!len[i] || npu[i] % RD_PAGE || off[i] % RD_PAGE || len[i] % RD_PAGE) return -EINVAL;
      if (len[i] > RD_DRAM_LIMIT || npu[i] > RD_DRAM_LIMIT - len[i]) return -EINVAL;
      if (len[i] > (uint64_t)st.st_size || off[i] > (uint64_t)st.st_size - len[i]) return -EINVAL;
      bytes += len[i];
      if (bytes > RD_MAX_CMD_BYTES) return -EINVAL;
    }
    first += counts[k];
  }
  if (first != ndesc) return -EINVAL;
  struct scatter_io_desc *desc = malloc(ndesc * sizeof *desc);
  if (!desc) return -ENOMEM;
  for (uint64_t i = 0; i < ndesc; i++) {
    uint64_t dev = NPU_DRAM_INWARD_ADDR + npu[i];
    desc[i] = (struct scatter_io_desc){.src = write ? off[i] : dev, .dst = write ? dev : off[i], .len = len[i]};
  }
  c->profile[0] = now_ns() - t0;
  /* The descriptor array must outlive every CQE: the driver may read it after submission (io-wq punt). */
  uint32_t prepared = 0; uint64_t done = 0; int err = 0;
  unsigned base = __atomic_load_n(c->sq_head, __ATOMIC_ACQUIRE), tail = *c->sq_tail;
  first = 0;
  for (;;) {
    while (!err && prepared < ncmd && prepared - done < c->depth) {
      unsigned slot = tail & *c->sq_mask;
      uint8_t *s = c->sqes + (size_t)slot * RD_SQE_BYTES;
      struct npu_pdma_dmabuf_io io = {.dmabuf_fd = fds[prepared], .desc_count = counts[prepared],
                                      .desc_addr = (uint64_t)(uintptr_t)(desc + first)};
      memset(s, 0, RD_SQE_BYTES);
      s[0] = RD_OP_URING_CMD;                                   /* opcode */
      memcpy(s + 4, &c->dev, 4);                                /* fd */
      uint32_t op = write ? NPU_PDMA_URING_CMD_DMABUF_WRITE : NPU_PDMA_URING_CMD_DMABUF_READ;
      memcpy(s + 8, &op, 4);                                    /* cmd_op */
      uint64_t user = (uint64_t)prepared + 1;
      memcpy(s + 32, &user, 8);                                 /* user_data */
      memcpy(s + 48, &io, sizeof io);                           /* cmd[] */
      c->sq_array[slot] = slot;
      first += counts[prepared]; prepared++; tail++;
      __atomic_store_n(c->sq_tail, tail, __ATOMIC_RELEASE);
    }
    unsigned unsubmitted = (base + prepared) - __atomic_load_n(c->sq_head, __ATOMIC_ACQUIRE);
    unsigned consumed = prepared - unsubmitted;
    done += reap(c, &err, completed);
    if (!err && prepared < ncmd && prepared - done < c->depth) continue;   /* a slot freed: fill it first */
    if (done == consumed && (err || prepared == ncmd) && !unsubmitted) break;
    if (err && done == consumed) {                              /* failed: never submit the rest */
      c->broken = 1;                                            /* prepared SQEs stay in the ring */
      break;
    }
    /* Wait only when a command is (or is about to be) in the kernel: never block on an empty ring. */
    /* The first submission is entered without waiting, so the profile separates submit from fence. */
    int first = !c->profile[1] && unsubmitted && !err;
    int r = enter(c->ring, err ? 0 : unsubmitted, first ? 0 : (done < consumed || (!err && unsubmitted) ? 1 : 0));
    if (first) c->profile[1] = now_ns() - t0;
    if (r < 0 && r != -EINTR && r != -EAGAIN && r != -EBUSY) {
      if (!err) err = r;
      unsubmitted = (base + prepared) - __atomic_load_n(c->sq_head, __ATOMIC_ACQUIRE);
      consumed = prepared - unsubmitted;
      /* The ring is unusable: wait for what the kernel already took, then refuse further work. */
      while (done < consumed) {
        int w = enter(c->ring, 0, 1);
        done += reap(c, &err, completed);
        if (w < 0 && w != -EINTR && w != -EAGAIN && w != -EBUSY && done < consumed) { c->broken = 1; return err; }  /* leak desc: the kernel may still read it */
      }
      c->broken = 1;
      break;
    }
  }
  free(desc);
  c->profile[2] = now_ns() - t0;
  return err;
}
