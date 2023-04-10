//
// Created by admin on 2022/9/21.
//

#ifndef CUDA_TEXTURE_REF_DEBUG_KERNELS_CUH
#define CUDA_TEXTURE_REF_DEBUG_KERNELS_CUH

#include <iostream>
#include "utils.cuh"
#include <cuda_runtime.h>


__global__ void sample_single_texture_1d_kernel(cudaTextureObject_t tex, float x, float4 *result);

float4 launch_sample_single_texture_1d_kernel(cudaTextureObject_t tex, float x);

__global__ void sample_single_texture_2d_kernel(cudaTextureObject_t tex, float x, float y, float4 *result);

float4 launch_sample_single_texture_2d_kernel(cudaTextureObject_t tex, float x, float y);

__global__ void sample_single_texture_3d_kernel(cudaTextureObject_t tex, float x, float y, float z, float4 *result);

float4 launch_sample_single_texture_3d_kernel(cudaTextureObject_t tex, float x, float y, float z);

template<typename T>
T access_data_on_device(T *data, int idx) {
    // we can just copy data up to that point
    T *host_data = new T[idx + 1];
    cudaMemcpy(host_data, data, sizeof(T) * (idx + 1), cudaMemcpyDeviceToHost);
    T ret = host_data[idx];
    delete[] host_data;
    return ret;
}

#endif //CUDA_TEXTURE_REF_DEBUG_KERNELS_CUH
