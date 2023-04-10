//
// Created by admin on 2022/9/21.
//

#include "debug_kernels.cuh"

__global__ void sample_single_texture_1d_kernel(cudaTextureObject_t tex, float x, float4 *result) {
    *result = tex1D<float4>(tex, x);
}

float4 launch_sample_single_texture_1d_kernel(cudaTextureObject_t tex, float x) {
    float4 *sampled = nullptr;
    cudaMalloc((void **) &sampled, sizeof(float4));
    sample_single_texture_1d_kernel<<<1, 1>>>(tex, x, sampled);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // copy result back to host variable
    float4 res;
    cudaMemcpy(&res, sampled, sizeof(float4), cudaMemcpyDeviceToHost);
    cudaFree(sampled);
    return res;
}

__global__ void
sample_single_texture_2d_kernel(cudaTextureObject_t tex, float x, float y, float4 *result) {
    *result = tex2D<float4>(tex, x, y);
}

float4 launch_sample_single_texture_2d_kernel(cudaTextureObject_t tex, float x, float y) {
    float4 *sampled = nullptr;
    cudaMalloc((void **) &sampled, sizeof(float4));
    sample_single_texture_2d_kernel<<<1, 1>>>(tex, x, y, sampled);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // copy result back to host variable
    float4 res;
    cudaMemcpy(&res, sampled, sizeof(float4), cudaMemcpyDeviceToHost);
    cudaFree(sampled);
    return res;
}

__global__ void sample_single_texture_3d_kernel(cudaTextureObject_t tex, float x, float y, float z, float4 *result) {
    *result = tex3D<float4>(tex, x, y, z);
}

float4 launch_sample_single_texture_3d_kernel(cudaTextureObject_t tex, float x, float y, float z) {
    float4 *sampled = nullptr;
    cudaMalloc((void **) &sampled, sizeof(float4));
    sample_single_texture_3d_kernel<<<1, 1>>>(tex, x, y, z, sampled);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // copy result back to host variable
    float4 res;
    cudaMemcpy(&res, sampled, sizeof(float4), cudaMemcpyDeviceToHost);
    cudaFree(sampled);
    return res;
}
