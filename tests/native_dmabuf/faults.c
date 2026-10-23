/* Execute the real submission/error loop with controlled ring memory and syscall failures. */
#define syscall test_syscall
#include "../../native/rngd_dmabuf.c"
#include <stdarg.h>
#include <assert.h>

static unsigned *test_head;
static unsigned *test_cq_tail;
static struct rd_cqe *test_cqe;
static int entered;
static int mode; /* 0: uncertain, 1: drained on error, 2: nothing submitted, 3: IOPOLL completes */
long test_syscall(long number, ...) {
  assert(number == __NR_io_uring_enter);
  va_list args;
  va_start(args, number);
  (void)va_arg(args, int); /* fd */
  unsigned submit = va_arg(args, unsigned), minimum = va_arg(args, unsigned), flags = va_arg(args, unsigned);
  va_end(args);
  if (!entered++ && mode != 2) *test_head = 1; /* consumed without completion */
  if ((mode == 1 && entered == 2) || (mode == 3 && !submit && !minimum && (flags & IORING_ENTER_GETEVENTS))) {
    *test_cqe = (struct rd_cqe){.user_data = 1, .res = -EIO};
    *test_cq_tail = 1;
  }
  errno = EIO;
  return -1;
}

int main(void) {
  unsigned sq_head = 0, sq_tail = 0, mask = 0, array = 0, cq_head = 0, cq_tail = 0;
  uint8_t sqes[RD_SQE_BYTES];
  struct rd_cqe cqes[1];
  struct rd_ctx ctx = {.dev = -1, .ring = -1, .depth = 1, .sq_head = &sq_head, .sq_tail = &sq_tail,
    .sq_mask = &mask, .sq_array = &array, .cq_head = &cq_head, .cq_tail = &cq_tail, .cq_mask = &mask,
    .sqes = sqes, .cqes = cqes};
  test_head = &sq_head;
  test_cq_tail = &cq_tail;
  test_cqe = cqes;
  FILE *file = tmpfile();
  assert(file && !ftruncate(fileno(file), 4096));
  int32_t fd = fileno(file);
  uint32_t count = 1;
  uint64_t npu = 0, off = 0, len = 4096, completed = 0;
  int rc = rd_run(&ctx, 0, 1, &fd, &count, 1, &npu, &off, &len, &completed);
  if (rc != -4096) { fprintf(stderr, "expected uncertain completion (-4096), got %d\n", rc); return 1; }
  assert(ctx.pending_desc && ctx.pending == 1);
  rd_close(&ctx); /* must leave the context and descriptors untouched while completion is unknown */
  assert(rd_quiesce(&ctx) == 0); /* failed enter alone is never evidence of a fence */
  assert(ctx.pending_desc && ctx.pending == 1);
  assert(rd_run(&ctx, 0, 1, &fd, &count, 1, &npu, &off, &len, &completed) == -4096);
  mode = 3; /* an IOPOLL ring requires GETEVENTS to publish the completed command's CQE */
  assert(rd_quiesce(&ctx) == 1);
  assert(!ctx.pending_desc && !ctx.pending);
  assert(rd_run(&ctx, 0, 1, &fd, &count, 1, &npu, &off, &len, &completed) == -EIO);
  for (mode = 1; mode <= 2; mode++) {
    ctx.broken = 0;
    sq_head = sq_tail = cq_head = cq_tail = 0;
    entered = 0;
    assert(rd_run(&ctx, 0, 1, &fd, &count, 1, &npu, &off, &len, &completed) == -EIO);
    assert(!ctx.pending_desc && !ctx.pending);
    assert(rd_quiesce(&ctx) == 1);
  }
  fclose(file);
  return 0;
}
