/* Test-only shape: these CPU fault tests do not validate the vendor UAPI ABI. */
#include <stdint.h>
#define NPU_PDMA_GET_CAPS 0
#define NPU_PDMA_CAPS_DMABUF 1
#define NPU_PDMA_URING_CMD_DMABUF_WRITE 1
#define NPU_PDMA_URING_CMD_DMABUF_READ 2
struct scatter_io_desc { uint64_t src, dst, len; };
struct npu_pdma_dmabuf_io { int32_t dmabuf_fd; uint32_t desc_count; uint64_t desc_addr; };
