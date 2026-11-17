// Small-row E4M3 GEMM with FP32 accumulation and BF16/FP16 output.
// Inputs and weights are contiguous [M,K] and [N,K]; scales are per row.
// Each CTA computes a 16x64 tile over one quarter of K. Two swizzled shared
// buffers pipeline 128-wide K tiles with cp.async; a second kernel combines
// the four partials before applying the input and weight scales. The MMA
// fragments below follow the PTX m16n8k32 row/column register layout.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <map>
#include <mutex>

namespace {

// A runtime SM120 GPU can execute PTX compiled from an older target. Query
// __CUDA_ARCH__ from the selected image instead of trusting the physical GPU.
__global__ void compiled_fp8_arch(int* arch) {
#ifdef __CUDA_ARCH__
  *arch = __CUDA_ARCH__;
#endif
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void async_copy16(uint8_t* dst,const uint8_t* src) {
  const unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(address),"l"(src):"memory");
}
__device__ __forceinline__ void load_tile(uint8_t* dst,const uint8_t* x,const uint8_t* w,
                                         int M,int K,int nbase,int kbase) {
  const int tid=threadIdx.x;
  const int ar=tid/8,ac=(tid%8)*16;
  const int source_row=ar<M?ar:ar%M;
  async_copy16(dst+ar*128+(ac^((ar&7)*16)),x+source_row*K+kbase+ac);
  #pragma unroll
  for(int j=0;j<4;++j) {
    const int index=tid+j*128,row=index/8,col=(index%8)*16;
    async_copy16(dst+2048+row*128+(col^((row&7)*16)),w+(nbase+row)*K+kbase+col);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ uint32_t shared_word(const uint8_t* p,int row,int col) {
  return *reinterpret_cast<const uint32_t*>(p+row*128+(col^((row&7)*16)));
}
__global__ void split_fp8(const uint8_t* __restrict__ x,const uint8_t* __restrict__ w,
                          float* __restrict__ partial,int M,int N,int K) {
  __shared__ __align__(16) uint8_t tiles[2][10240];
  // Four warps cover a 16x64 output tile; each warp owns 16 columns.
  const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  const int group=lane>>2,t=lane&3;
  const int nbase=blockIdx.x*64,begin=blockIdx.y*(K/4),chunks=K/(4*128);
  float d[2][4]={{0,0,0,0},{0,0,0,0}};
  load_tile(tiles[0],x,w,M,K,nbase,begin);
  asm volatile("cp.async.wait_group 0;" ::: "memory");__syncthreads();
  for(int tile=0;tile<chunks;++tile) {
    if(tile+1<chunks)load_tile(tiles[(tile+1)&1],x,w,M,K,nbase,begin+(tile+1)*128);
    const uint8_t* current=tiles[tile&1];
    #pragma unroll
    for(int kk=0;kk<128;kk+=32) {
      uint32_t a[4]={shared_word(current,group,kk+4*t),shared_word(current,group+8,kk+4*t),
                     shared_word(current,group,kk+16+4*t),shared_word(current,group+8,kk+16+4*t)};
      #pragma unroll
      for(int j=0;j<2;++j) {
        const int row=warp*16+8*j+group;
        mma(d[j],a,shared_word(current+2048,row,kk+4*t),shared_word(current+2048,row,kk+16+4*t));
      }
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");__syncthreads();
  }
  #pragma unroll
  for(int j=0;j<2;++j) {
    const int col=nbase+warp*16+8*j+2*t;
    if(group<M)*reinterpret_cast<float2*>(partial+blockIdx.y*M*N+group*N+col)=make_float2(d[j][0],d[j][1]);
    if(group+8<M)*reinterpret_cast<float2*>(partial+blockIdx.y*M*N+(group+8)*N+col)=make_float2(d[j][2],d[j][3]);
  }
}

#else
// Keep older-target builds valid, but never silently produce zero projections.
// The host preflight rejects this image before this defensive stub can launch.
__global__ void split_fp8(const uint8_t*, const uint8_t*, float*, int, int, int) {
  asm volatile("trap;");
}
#endif

template<typename T> __global__ void finish_fp8(const float* __restrict__ partial,
                      const float* __restrict__ sa,const float* __restrict__ sb,
                      T* __restrict__ out,int M,int N) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<M*N) {
    const int stride=M*N;
    const float value=(partial[i]+partial[i+stride])+(partial[i+2*stride]+partial[i+3*stride]);
    const float result=value*(sa[i/N]*sb[i%N]);
    if constexpr(std::is_same_v<T,__nv_bfloat16>) out[i]=__float2bfloat16_rn(result);
    else out[i]=__float2half_rn(result);
  }
}

}  // namespace

// Resolve once per device during module preparation, outside CUDA capture.
// Checking the selected image also catches old PTX JIT-compiled on a newer GPU.
int64_t fp8_compiled_arch(int64_t device) {
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
  static std::mutex mutex;
  static std::map<int64_t, int64_t> cache;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto found = cache.find(device);
    if (found != cache.end()) return found->second;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaStreamCaptureStatus status;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
  TORCH_CHECK(status == cudaStreamCaptureStatusNone,
              "Initialize FP8 MMA support before CUDA graph capture");
  auto value = torch::empty({1}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt));
  compiled_fp8_arch<<<1, 1, 0, stream>>>(value.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  int arch = 0;
  C10_CUDA_CHECK(cudaMemcpyAsync(&arch, value.data_ptr<int>(), sizeof(arch), cudaMemcpyDeviceToHost, stream));
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  // Do not hold the cache mutex while waiting on a CUDA stream.
  std::lock_guard<std::mutex> lock(mutex);
  cache.emplace(device, arch);
  return arch;
}

bool fp8_mma_supported(int64_t device) {
  return fp8_compiled_arch(device) >= 890;
}

// Small-row E4M3 GEMM: four K splits, asynchronous shared-memory staging.
torch::Tensor fp8_mm(torch::Tensor x,torch::Tensor w,torch::Tensor sa,torch::Tensor sb,bool bf16=true) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && sa.is_cuda() && sb.is_cuda(),"CUDA tensors required");
  TORCH_CHECK(x.device()==w.device() && x.device()==sa.device() && x.device()==sb.device(),"device mismatch");
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.dim()==2 && w.dim()==2 && x.scalar_type()==at::kFloat8_e4m3fn && w.scalar_type()==at::kFloat8_e4m3fn,"E4M3 matrices required");
  const int M=x.size(0),K=x.size(1),N=w.size(0);
  TORCH_CHECK(M>=1 && M<=16 && K>=512 && K%512==0 && N>=64 && N%64==0 && w.size(1)==K,"unsupported dimensions");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && sa.is_contiguous() && sb.is_contiguous(),"contiguous buffers required");
  TORCH_CHECK(sa.scalar_type()==at::kFloat && sb.scalar_type()==at::kFloat && sa.numel()==M && sb.numel()==N,"FP32 per-row scales required");
  auto prop=at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major*10+prop->minor>=89,"FP8 MMA requires SM89+");
  TORCH_CHECK(fp8_mma_supported(x.get_device()),
              "Eole FP8 MMA was compiled for a pre-SM89 target; rebuild with an SM89+ "
              "CUDA target or use PyTorch scaled GEMM");
  auto partial=torch::empty({4,M,N},x.options().dtype(at::kFloat));
  auto out=torch::empty({M,N},x.options().dtype(bf16?at::kBFloat16:at::kHalf));
  auto stream=at::cuda::getCurrentCUDAStream();
  split_fp8<<<dim3(N/64,4),128,0,stream>>>(reinterpret_cast<const uint8_t*>(x.data_ptr()),reinterpret_cast<const uint8_t*>(w.data_ptr()),partial.data_ptr<float>(),M,N,K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if(bf16)finish_fp8<<<(M*N+255)/256,256,0,stream>>>(partial.data_ptr<float>(),sa.data_ptr<float>(),sb.data_ptr<float>(),reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),M,N);
  else finish_fp8<<<(M*N+255)/256,256,0,stream>>>(partial.data_ptr<float>(),sa.data_ptr<float>(),sb.data_ptr<float>(),reinterpret_cast<__half*>(out.data_ptr()),M,N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
