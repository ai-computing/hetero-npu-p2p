/* Completion-fenced RNGD PDMA <-> imported DMA-BUF (e.g. CUDA memory exported by the NVIDIA open kernel module).
 * CLI: device_path read|write plan_file descriptors_per_command queue_depth reps
 *   read : RNGD DRAM -> DMA-BUF (device -> peer)     write: DMA-BUF -> RNGD DRAM (peer -> device)
 * Plan rows: inherited_fd npu_dram_offset dmabuf_offset byte_count  (4 KiB aligned, <= 2 MiB). A command holds up to
 * descriptors_per_command consecutive rows of one FD (a new command starts where the FD changes).
 * No host data buffers and no host-copy fallback. Prints one JSON receipt; rc 0 only if every CQE completed.
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
struct item { int fd; uint64_t npu,off,len,dev,ino; };
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int rejected(int line){fprintf(stderr,"plan validation rejected at line %d\n",line);return 2;}
static int number(const char *s,uint64_t *v){char *e;errno=0;*v=strtoull(s,&e,0);return !errno&&*s&&!*e&&*s!='-';}
int main(int argc,char **argv){
 unsigned int device_id; char extra_path; uint64_t batch,qd,reps;
 if(argc!=7||sscanf(argv[1],"/dev/rngd/npu%uch0%c",&device_id,&extra_path)!=1)return rejected(__LINE__);
 char expected[64];snprintf(expected,sizeof(expected),"/dev/rngd/npu%uch0",device_id);
 if(strcmp(expected,argv[1]))return rejected(__LINE__);
 int write_dev=!strcmp(argv[2],"write");
 if(!write_dev&&strcmp(argv[2],"read"))return rejected(__LINE__);
 if(!number(argv[4],&batch)||!batch||batch>64||!number(argv[5],&qd)||!qd||qd>16||!number(argv[6],&reps)||!reps||reps>100)
  return rejected(__LINE__);
 const uint64_t limit=48ULL*1024*1024*1024-512ULL*1024*1024;
 FILE *f=fopen(argv[3],"r");if(!f)return rejected(__LINE__);
 size_t cap=1024,count=0;struct item *items=malloc(cap*sizeof *items);char line[256];uint64_t total=0;
 while(fgets(line,sizeof(line),f)){
  struct item v;char extra;struct stat st;
  if(count==8192||sscanf(line,"%d %"SCNu64" %"SCNu64" %"SCNu64" %c",&v.fd,&v.npu,&v.off,&v.len,&extra)!=4){fclose(f);return rejected(__LINE__);}
  if(v.fd<0||!v.len||v.len>2097152||v.npu%4096||v.off%4096||v.len%4096||v.npu>limit-v.len||fstat(v.fd,&st)||st.st_size<0
     ||v.len>(uint64_t)st.st_size||v.off>(uint64_t)st.st_size-v.len){fclose(f);return rejected(__LINE__);}
  v.dev=(uint64_t)st.st_dev;v.ino=(uint64_t)st.st_ino;
  if(count==cap){cap*=2;items=realloc(items,cap*sizeof *items);}
  items[count++]=v;total+=v.len;
 }
 if(ferror(f)||!count){fclose(f);return rejected(__LINE__);}fclose(f);
 // writes into one destination must not overlap (RNGD ranges for write, DMA-BUF ranges for read)
 for(size_t i=0;i<count;i++)for(size_t j=0;j<i;j++){
  uint64_t a=write_dev?items[i].npu:items[i].off,b=write_dev?items[j].npu:items[j].off;
  int same=write_dev||(items[i].dev==items[j].dev&&items[i].ino==items[j].ino);
  if(same&&a<b+items[j].len&&b<a+items[i].len)return rejected(__LINE__);
 }
 size_t *start=malloc((count+1)*sizeof *start),ncmd=0;
 for(size_t i=0,run=0;i<count;i++,run++){if(!i||items[i].fd!=items[i-1].fd||run==batch){start[ncmd++]=i;run=0;}}
 start[ncmd]=count;
 struct scatter_io_desc *desc=calloc(count,sizeof *desc);
 for(size_t i=0;i<count;i++){
  uint64_t npu=NPU_DRAM_INWARD_ADDR+items[i].npu;
  desc[i]=(struct scatter_io_desc){.src=write_dev?items[i].off:npu,.dst=write_dev?npu:items[i].off,.len=items[i].len};
 }
 int fd=open(argv[1],O_RDWR|O_CLOEXEC);if(fd<0){perror("open");return 3;}
 int caps=ioctl(fd,NPU_PDMA_GET_CAPS);if(caps<0||!(caps&NPU_PDMA_CAPS_DMABUF)){close(fd);return 3;}
 struct io_uring ring;struct io_uring_params params={0};params.flags=IORING_SETUP_SQE128;
 int ret=io_uring_queue_init_params((unsigned)qd*2,&ring,&params);if(ret<0){close(fd);return 4;}
 size_t completed=0;int result=0;double first=0,best=1e30;
 printf("{\"direction\":\"%s\",\"seconds\":[",argv[2]);
 for(uint64_t r=0;r<reps&&!result;r++){
  size_t sent=0,done=0;double t0=now();
  while(done<ncmd&&!result){
   while(sent<ncmd&&sent-done<qd){
    size_t i=start[sent],n=start[sent+1]-start[sent];
    struct npu_pdma_dmabuf_io command={.dmabuf_fd=items[i].fd,.desc_count=(uint32_t)n,.desc_addr=(uint64_t)(uintptr_t)(desc+i)};
    struct io_uring_sqe *sqe=io_uring_get_sqe(&ring);
    io_uring_prep_rw(IORING_OP_URING_CMD,sqe,fd,NULL,0,0);
    sqe->cmd_op=write_dev?NPU_PDMA_URING_CMD_DMABUF_WRITE:NPU_PDMA_URING_CMD_DMABUF_READ;
    memcpy(sqe->cmd,&command,sizeof(command));sqe->user_data=sent+1;sent++;
   }
   if((ret=io_uring_submit(&ring))<0){result=ret;break;}
   struct io_uring_cqe *cqe=NULL;
   do{ret=io_uring_wait_cqe(&ring,&cqe);}while(ret==-EINTR);
   if(ret<0){result=ret;break;}
   if(cqe->res)result=cqe->res;
   io_uring_cqe_seen(&ring,cqe);done++;
  }
  // drain what is still in flight before reporting (never leave commands outstanding)
  while(done<sent){struct io_uring_cqe *cqe=NULL;do{ret=io_uring_wait_cqe(&ring,&cqe);}while(ret==-EINTR);
   if(ret<0){result=ret;break;} if(cqe->res&&!result)result=cqe->res; io_uring_cqe_seen(&ring,cqe);done++;}
  double dt=now()-t0;if(!r)first=dt;if(dt<best)best=dt;
  printf("%s%.9f",r?",":"",dt);if(!result)completed+=done;
 }
 printf("],\"cqe_result\":%d,\"completed\":%zu,\"requested\":%zu,\"bytes_requested\":%"PRIu64",\"first_seconds\":%.9f,\"best_seconds\":%.9f}\n",
        result,completed,ncmd*(size_t)reps,total,first,best);
 io_uring_queue_exit(&ring);close(fd);free(desc);free(items);free(start);
 return result==0&&completed==ncmd*(size_t)reps?0:5;
}
