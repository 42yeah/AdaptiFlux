#ifndef YYLVV_UTILS_CUH
#define YYLVV_UTILS_CUH

#include <cmath>
#include <glm/glm.hpp>

#define CHECK_CUDA_ERROR(val) check_cuda_result((val), (#val), __FILE__, __LINE__)

template<typename T>
void check_cuda_result(T result, const char *const func, const char *const file, int line) {
    if (result) {
        std::cerr << "CUDA error at " << file << ":" << line << " with code=" << ((unsigned int) result) << ", " << cudaGetErrorString(result) << ", \"" << func << "\"?" << std::endl;
    }
}

inline float vector_length(float x, float y, float z) {
    return sqrtf(x * x + y * y + z * z);
}

__device__ inline float3 float3_scale(float scale, float3 f) {
    return make_float3(scale * f.x, scale * f.y, scale * f.z);
}

__device__ inline float4 float4_scale(float scale, float4 f) {
    return make_float4(scale * f.x, scale * f.y, scale * f.z, scale * f.w);
}

__device__ inline float3 float3_normalize(float4 vec) {
    return make_float3(vec.x / vec.w, vec.y / vec.w, vec.z / vec.w);
}

__device__ inline float3 float3_add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline void set_float3(float *float_arr, int offset, float3 fl3) {
    float_arr[offset + 0] = fl3.x;
    float_arr[offset + 1] = fl3.y;
    float_arr[offset + 2] = fl3.z;
}

__device__ inline void set_vec3(float *float_arr, int offset, glm::vec3 vec3) {
    float_arr[offset + 0] = vec3.x;
    float_arr[offset + 1] = vec3.y;
    float_arr[offset + 2] = vec3.z;    
}

__device__ inline void set_xyz(float *float_arr, int offset, float x, float y, float z) {
    float_arr[offset + 0] = x;
    float_arr[offset + 1] = y;
    float_arr[offset + 2] = z;
}

__device__ inline float4 operator+(const float4 &a, const float4 &b) 
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ inline glm::vec3 float4_to_vec3(float4 vec)
{
    return glm::vec3(vec.x, vec.y, vec.z);
}

__device__ inline float4 sample_texture(cudaTextureObject_t tex, const glm::vec3 &p)
{
    return tex3D<float4>(tex, p.x, p.y, p.z);
}

inline bool ends_with(const std::string &str, const std::string &suffix)
{
    if (str.size() < suffix.size())
    {
        return false;
    }

    return str.substr(str.size() - suffix.size(), suffix.size()) == suffix;
}

#endif
