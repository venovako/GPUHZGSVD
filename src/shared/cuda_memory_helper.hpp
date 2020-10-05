#ifndef CUDA_MEMORY_HELPER_HPP
#define CUDA_MEMORY_HELPER_HPP

#include "cuda_helper.hpp"

template <typename T>
T *allocHostMtx(size_t &ldm, const size_t m, const size_t n, const bool f) throw()
{
  T *ret = static_cast<T*>(NULL);
  const size_t b = ldm * ((f ? n : m) * sizeof(T));

  if (b && (ldm >= (f ? m : n)))
    CUDA_CALL(cudaHostAlloc(&ret, b, cudaHostAllocPortable));

  return static_cast<T*>(ret ? memset(ret, 0, b) : NULL);
}

template <typename T>
T *allocHostVec(const size_t m) throw()
{
  T *ret = static_cast<T*>(NULL);
  const size_t b = m * sizeof(T);

  if (b)
    CUDA_CALL(cudaHostAlloc(&ret, b, cudaHostAllocPortable));

  return static_cast<T*>(ret ? memset(ret, 0, b) : NULL);
}

template <typename T>
T *allocDeviceMtx(size_t &ldm, const size_t m, const size_t n, const bool f, const cudaStream_t s = 0) throw()
{
  T *ret = static_cast<T*>(NULL);

  if (ldm && (f ? n : m) && (ldm >= (f ? m : n))) {
    size_t pitch = static_cast<size_t>(0u);
    CUDA_CALL(cudaMallocPitch(&ret, &pitch, ldm * sizeof(T), (f ? n : m)));
    ldm = pitch / sizeof(T);
    if (ret) {
      CUDA_CALL(cudaMemset2DAsync(ret, pitch, 0, pitch, (f ? n : m), s));
    }
  }

  return ret;
}

template <typename T>
T *allocDeviceVec(const size_t m, const cudaStream_t s = 0) throw()
{
  T *ret = static_cast<T*>(NULL);
  const size_t b = m * sizeof(T);

  if (b) {
    CUDA_CALL(cudaMalloc(&ret, b));
    if (ret) {
      CUDA_CALL(cudaMemsetAsync(ret, 0, b, s));
    }
  }

  return ret;
}

#endif /* !CUDA_MEMORY_HELPER_HPP */
