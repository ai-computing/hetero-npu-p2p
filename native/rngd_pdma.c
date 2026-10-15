/* Sequential, completion-fenced RNGD1 PE0 -> exported ATOM cache writes.
 * CLI: device_path source_base source_bytes plan_file descriptors_per_command.
 * Plan rows: inherited_fd source_offset destination_offset byte_count.
 * No host data buffers and no host-copy fallback. Source is owned by caller.
 */
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <time.h>
#include <liburing.h>
#include "npu_pdma_uapi.h"
#include "npu_address_map.h"
struct item { int fd; uint64_t src,dst,len,dev,ino; };
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int number(const char *s,uint64_t *v){char *e;errno=0;*v=strtoull(s,&e,0);return !errno&&*s&&!*e&&*s!='-';}
static int rejected(int line){fprintf(stderr,"plan validation rejected at line %d\n",line);return 2;}
int main(int argc,char **argv){
 unsigned int device_id; char extra_path;
 if(argc!=6||sscanf(argv[1],"/dev/rngd/npu%uch0%c",&device_id,&extra_path)!=1)return rejected(__LINE__);
 char expected_path[64];snprintf(expected_path,sizeof(expected_path),"/dev/rngd/npu%uch0",device_id);
 if(strcmp(expected_path,argv[1]))return rejected(__LINE__);
 uint64_t base,source_size,batch;
 if(argc!=6||!number(argv[2],&base)||!number(argv[3],&source_size)||!number(argv[5],&batch)||!batch||batch>8)return rejected(__LINE__);
 const uint64_t limit=48ULL*1024*1024*1024-512ULL*1024*1024;
 if(!base||base%64||!source_size||source_size>64ULL*1024*1024||source_size%4096||source_size>limit||base>limit-source_size)return rejected(__LINE__);
 FILE *f=fopen(argv[4],"r");if(!f)return rejected(__LINE__);
 struct item items[512];size_t count=0;char line[256];uint64_t total=0;
 while(fgets(line,sizeof(line),f)){
  struct item v;char extra;
  if(count==512||sscanf(line,"%d %"SCNu64" %"SCNu64" %"SCNu64" %c",&v.fd,&v.src,&v.dst,&v.len,&extra)!=4){fclose(f);return rejected(__LINE__);}
  struct stat st;
  if(v.fd<0||!v.len||v.len>2097152||v.src%4096||v.dst%4096||v.len%4096||v.len>source_size||v.src>source_size-v.len||fstat(v.fd,&st)||st.st_size<0||v.len>(uint64_t)st.st_size||v.dst>(uint64_t)st.st_size-v.len){fclose(f);return rejected(__LINE__);}
  v.dev=(uint64_t)st.st_dev;v.ino=(uint64_t)st.st_ino;
  for(size_t j=0;j<count;j++)if(items[j].dev==v.dev&&items[j].ino==v.ino&&v.dst<items[j].dst+items[j].len&&items[j].dst<v.dst+v.len){fclose(f);return rejected(__LINE__);}
  items[count++]=v;total+=v.len;
 }
 if(ferror(f)||!count||count%batch){fclose(f);return rejected(__LINE__);}fclose(f);
 for(size_t i=0;i<count;i+=batch)
  for(size_t j=1;j<batch;j++)if(items[i+j].fd!=items[i].fd)return rejected(__LINE__);
 int fd=open(argv[1],O_RDWR|O_CLOEXEC);if(fd<0){perror("open");return 3;}
 int caps=ioctl(fd,NPU_PDMA_GET_CAPS);if(caps<0||!(caps&NPU_PDMA_CAPS_DMABUF)){close(fd);return 3;}
 struct io_uring ring;struct io_uring_params params={0};params.flags=IORING_SETUP_SQE128;
 int ret=io_uring_queue_init_params(2,&ring,&params);if(ret<0){close(fd);return 4;}
 size_t completed=0;double start=now();
 for(size_t i=0;i<count;i+=batch){
  struct item v=items[i];
  struct scatter_io_desc desc[8];
  for(size_t j=0;j<batch;j++){struct item a=items[i+j];desc[j]=(struct scatter_io_desc){.src=NPU_DRAM_INWARD_ADDR+base+a.src,.dst=a.dst,.len=a.len};}
  struct npu_pdma_dmabuf_io command={.dmabuf_fd=v.fd,.desc_count=(uint32_t)batch,.desc_addr=(uint64_t)(uintptr_t)desc};
  struct io_uring_sqe *sqe=io_uring_get_sqe(&ring);
  io_uring_prep_rw(IORING_OP_URING_CMD,sqe,fd,NULL,0,0);sqe->cmd_op=NPU_PDMA_URING_CMD_DMABUF_READ;
  memcpy(sqe->cmd,&command,sizeof(command));sqe->user_data=i+1;
  ret=io_uring_submit(&ring);if(ret!=1){if(ret>=0)ret=-EIO;break;}
  struct io_uring_cqe *cqe=NULL;
  do {ret=io_uring_wait_cqe(&ring,&cqe);}while(ret==-EINTR);
  if(ret<0)break;
  ret=cqe->res;if(cqe->user_data!=i+1)ret=-EIO;
  io_uring_cqe_seen(&ring,cqe);if(ret)break;completed++;
 }
 printf("{\"cqe_result\":%d,\"completed\":%zu,\"requested\":%zu,\"bytes_requested\":%"PRIu64",\"dma_seconds\":%.9f}\n",ret,completed,count/batch,total,now()-start);
 io_uring_queue_exit(&ring);close(fd);return ret==0&&completed==count/batch?0:5;
}
