#ifndef YYLVV_CUH
#define YYLVV_CUH

#include <NRRD.h>
#include <VectorField.h> // for BBox
#include <iostream>
#include <cuda_runtime.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "utils.cuh"
#include "PlainText.h"

struct CUDATexture3D {
    BBox get_bounding_box() const;

    cudaTextureObject_t texture;
    cudaArray_t array;
    cudaExtent extent;
    float longest_vector;
    float average_vector;
};

struct YYLVVRes {
    CUDATexture3D vf_tex; // CUDA vector field texture
    GLFWwindow *window; // YYLVV visualizer window
};

bool initialize_yylvv_contents(int argc, char *argv[], YYLVVRes &res);
GLFWwindow *create_yylvv_window(int width, int height, const std::string &title);
bool nrrd_to_3d_texture(NRRD &nrrd, CUDATexture3D &ret_tex);
bool plain_text_to_3d_texture(PlainText &plain_text, CUDATexture3D &ret_tex);
bool free_yylvv_resources(YYLVVRes &res);

// === IMPLEMENTATIONS ===

#ifdef YYLVV_IMPL

bool initialize_yylvv_contents(int argc, char *argv[], YYLVVRes &res) {
    if (argc != 2) {
        std::cerr << "Wrong number of arguments: " << argc << "?" << std::endl;
        return false;
    }

    std::string file_name = argv[1];
    if (ends_with(file_name, ".nrrd"))
    {
        NRRD nrrd;
        if (!nrrd.load_from_file(file_name))
        {
            std::cerr << "Cannot load NRRD?" << std::endl;
            return false;
        }
        
        std::cout << "NRRD loaded. Size: " << nrrd.sizes[0] << ", " << nrrd.sizes[1] << ", " << nrrd.sizes[2] << ", " << nrrd.sizes[3] << std::endl;
        std::cout << "Loading vector field into CUDA 3D texture..." << std::endl;
        if (!nrrd_to_3d_texture(nrrd, res.vf_tex))
        {
            std::cerr << "Cannot transform NRRD into 3D texture?" << std::endl;
            return false;
        }
    }
    else if (ends_with(file_name, ".txt"))
    {
        PlainText plain_text;

        if (!plain_text.load_from_file(file_name))
        {
            std::cerr << "Failed to load: " << file_name << "?" << std::endl;
            return false;
        }
        
        if (!plain_text_to_3d_texture(plain_text, res.vf_tex))
        {
            std::cerr << "Cannot transform plain text data into 3D texture?" << std::endl;
            return false;
        }
    }
    else
    {
        std::cerr << "Mysterious file type: " << file_name << "?" << std::endl;
        return false;
    }

    std::cout << "Creating YYLVV window and OpenGL context." << std::endl;
    res.window = create_yylvv_window(1024, 768, "YYLVV visualizer");
    if (!res.window) {
        std::cerr << "Cannot create YYLVV window?" << std::endl;
        return false;
    }
    return true;
}

bool nrrd_to_3d_texture(NRRD &nrrd, CUDATexture3D &ret_tex) {
    std::unique_ptr<float4[]> vf_float4 = std::make_unique<float4[]>(nrrd.sizes[1] * nrrd.sizes[2] * nrrd.sizes[3]);
    float longest = 0.0f;
    double sum = 0.0;
    for (int z = 0; z < nrrd.sizes[3]; z++) {
        for (int y = 0; y < nrrd.sizes[2]; y++) {
            for (int x = 0; x < nrrd.sizes[1]; x++) {
                int idx = z * nrrd.sizes[2] * nrrd.sizes[1] + y * nrrd.sizes[1] + x;
                int nrrd_idx = idx * 3; // pitched by 3
                float vx = nrrd.raw_data[nrrd_idx + 0],
                    vy = nrrd.raw_data[nrrd_idx + 1],
                    vz = nrrd.raw_data[nrrd_idx + 2];
                float vl = vector_length(vx, vy, vz);
                if (longest < vl) {
                    longest = vl;
                }
                sum += vl;
                vf_float4[idx] = make_float4(vx, vy, vz, vl);
            }
        }
    }
    std::cout << "Allocating 3D CUDA array for vector field..." << std::endl;
    size_t vf_float4_size_in_bytes = nrrd.sizes[1] * nrrd.sizes[2] * nrrd.sizes[3] * sizeof(float4);
    size_t w_pitch = nrrd.sizes[1] * sizeof(float4);
    // malloc & copy 3D array to data
    cudaArray_t vf_float4_cuda = nullptr;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
    cudaExtent extent = make_cudaExtent(nrrd.sizes[1], nrrd.sizes[2], nrrd.sizes[3]);
    CHECK_CUDA_ERROR(cudaMalloc3DArray(&vf_float4_cuda, &desc, extent, 0));
    cudaMemcpy3DParms vf_float4_copy_params = {0};
    vf_float4_copy_params.srcPtr = make_cudaPitchedPtr((void *) vf_float4.get(), w_pitch, nrrd.sizes[1], nrrd.sizes[2]);
    vf_float4_copy_params.dstArray = vf_float4_cuda;
    vf_float4_copy_params.extent = extent;
    vf_float4_copy_params.kind = cudaMemcpyHostToDevice;
    CHECK_CUDA_ERROR(cudaMemcpy3D(&vf_float4_copy_params));

    std::cout << "Binding texture to 3D CUDA array..." << std::endl;
    cudaResourceDesc rdesc;
    std::memset(&rdesc, 0, sizeof(cudaResourceDesc));
    {
        rdesc.resType = cudaResourceTypeArray;
        rdesc.res.array.array = vf_float4_cuda;
    }
    cudaTextureDesc tdesc;
    std::memset(&tdesc, 0, sizeof(cudaTextureDesc));
    {
        tdesc.addressMode[0] = cudaAddressModeClamp;
        tdesc.addressMode[1] = cudaAddressModeClamp;
        tdesc.addressMode[2] = cudaAddressModeClamp;
        tdesc.filterMode = cudaFilterModeLinear;
        tdesc.readMode = cudaReadModeElementType;
        tdesc.normalizedCoords = 0;
    }
    cudaTextureObject_t vf_tex;
    CHECK_CUDA_ERROR(cudaCreateTextureObject(&vf_tex, &rdesc, &tdesc, nullptr));
    std::cout << "Texture creation complete." << std::endl;

    ret_tex.texture = vf_tex;
    ret_tex.array = vf_float4_cuda;
    ret_tex.extent = extent;
    ret_tex.longest_vector = longest;
    ret_tex.average_vector = (float) (sum / (nrrd.sizes[1] * nrrd.sizes[2] * nrrd.sizes[3]));
    return true;
}

bool plain_text_to_3d_texture(PlainText &plain_text, CUDATexture3D &ret_tex)
{
    if (!plain_text.raw_data)
    {
        std::cerr << "No input plain text vector field data?" << std::endl;
        return false;
    }

    std::unique_ptr<float4[]> vf_float4 = std::make_unique<float4[]>(plain_text.sizes.x * plain_text.sizes.y * plain_text.sizes.z);
    float longest = 0.0f;
    double sum = 0.0;

    for (int z = 0; z < plain_text.sizes.z; z++) 
    {
        for (int y = 0; y < plain_text.sizes.y; y++) 
        {
            for (int x = 0; x < plain_text.sizes.x; x++) 
            {
                int idx = z * plain_text.sizes.y * plain_text.sizes.x + y * plain_text.sizes.x + x;
                int pt_idx = idx * 3; // pitched by 3
                float vx = plain_text.raw_data[pt_idx + 0],
                    vy = plain_text.raw_data[pt_idx + 1],
                    vz = plain_text.raw_data[pt_idx + 2];
                float vl = vector_length(vx, vy, vz);
                
                if (longest < vl) 
                {
                    longest = vl;
                }
                sum += vl;

                // std::cout << x << ", " << y << ", " << z << " is " << vl << " long" << std::endl;
                vf_float4[idx] = make_float4(vx, vy, vz, vl);
            }
        }
    }

    std::cout << "Allocating CUDA array: (" << plain_text.sizes.x << ", " << 
        plain_text.sizes.y << ", " << plain_text.sizes.z << 
        ") with channel size of " << sizeof(float4) << std::endl;

    cudaArray_t vf_float4_cuda = nullptr;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
    cudaExtent extent = make_cudaExtent(plain_text.sizes.x, plain_text.sizes.y, plain_text.sizes.z);

    CHECK_CUDA_ERROR(cudaMalloc3DArray(&vf_float4_cuda, &desc, extent, 0));

    const int w_pitch = plain_text.sizes.x * sizeof(float4);
    cudaMemcpy3DParms vf_float4_copy_params = {0};

    vf_float4_copy_params.srcPtr = make_cudaPitchedPtr((void *) vf_float4.get(), w_pitch, plain_text.sizes.x, plain_text.sizes.y);
    vf_float4_copy_params.dstArray = vf_float4_cuda;
    vf_float4_copy_params.extent = extent;
    vf_float4_copy_params.kind = cudaMemcpyHostToDevice;

    CHECK_CUDA_ERROR(cudaMemcpy3D(&vf_float4_copy_params));

    cudaResourceDesc rdesc;
    std::memset(&rdesc, 0, sizeof(cudaResourceDesc));
    rdesc.resType = cudaResourceTypeArray;
    rdesc.res.array.array = vf_float4_cuda;

    cudaTextureDesc tdesc;
    std::memset(&tdesc, 0, sizeof(cudaTextureDesc));
    tdesc.addressMode[0] = cudaAddressModeClamp;
    tdesc.addressMode[1] = cudaAddressModeClamp;
    tdesc.addressMode[2] = cudaAddressModeClamp;
    tdesc.filterMode = cudaFilterModeLinear;
    tdesc.readMode = cudaReadModeElementType;
    tdesc.normalizedCoords = 0;

    cudaTextureObject_t vf_tex;
    CHECK_CUDA_ERROR(cudaCreateTextureObject(&vf_tex, &rdesc, &tdesc, nullptr));

    ret_tex.texture = vf_tex;
    ret_tex.array = vf_float4_cuda;
    ret_tex.extent = extent;
    ret_tex.longest_vector = longest;
    ret_tex.average_vector = (float) (sum / (plain_text.sizes.x * plain_text.sizes.y * plain_text.sizes.z));

    return true;
}

bool free_yylvv_resources(YYLVVRes &res) {
    std::cout << "Freeing 3D texture and array." << std::endl;
    CHECK_CUDA_ERROR(cudaDestroyTextureObject(res.vf_tex.texture));
    CHECK_CUDA_ERROR(cudaFreeArray(res.vf_tex.array));
    glfwDestroyWindow(res.window);
    return true;
}

GLFWwindow *create_yylvv_window(int width, int height, const std::string &title) {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    GLFWwindow *window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window) {
        return nullptr;
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGL()) {
        std::cerr << "Failed to load GLAD GL?" << std::endl;
        glfwDestroyWindow(window);
        return nullptr;
    }
    return window;
}

BBox CUDATexture3D::get_bounding_box() const {
    BBox bbox;
    bbox.enclose(glm::vec3(0.0f, 0.0f, 0.0f));
    bbox.enclose(glm::vec3((float) extent.width,
                           (float) extent.height,
                           (float) extent.depth));
    return bbox;
}

#endif
#endif
