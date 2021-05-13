#include <cstdio>

__global__ void thread(float *a) {
  a[threadIdx.x] = threadIdx.x;
}

int main(void) {
  const int N = 4;
  // 共有メモリモデル
  // float *a;
  // cudaMallocManaged(&a, N*sizeof(float));

  // 分散メモリモデル
  float *a;
  float *b=(float*)malloc(N*sizeof(float));
  cudaMalloc(&a,N*sizeof(float));
  // cudaMemcpy(b,a,N*sizeof(float),cudaMemcpyHostToDevice);

  thread<<<1,N>>>(a);
  // cudaDeviceSynchronize();
  cudaMemcpy(a,b,N*sizeof(float),cudaMemcpyDeviceToHost);
  for (int i=0; i<N; i++)
    printf("%d %g\n",i,b[i]);
  cudaFree(a);
}