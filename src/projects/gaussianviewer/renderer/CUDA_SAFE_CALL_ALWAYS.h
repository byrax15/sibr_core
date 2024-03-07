#define CUDA_SAFE_CALL_ALWAYS(A)              \
    A;                                        \
    cudaDeviceSynchronize();                  \
    if (cudaPeekAtLastError() != cudaSuccess) \
        SIBR_ERR << cudaGetErrorString(cudaGetLastError());

#if DEBUG || _DEBUG
#define CUDA_SAFE_CALL(A) CUDA_SAFE_CALL_ALWAYS(A)
#else
#define CUDA_SAFE_CALL(A) A
#endif