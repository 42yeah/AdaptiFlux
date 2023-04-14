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
    std::string vf_name;
    CUDATexture3D vf_tex; // CUDA vector field texture
    GLFWwindow *window; // YYLVV visualizer window
};

bool initialize_yylvv_contents(int argc, char *argv[], YYLVVRes &res);
GLFWwindow *create_yylvv_window(int width, int height, const std::string &title);
bool nrrd_to_3d_texture(NRRD &nrrd, CUDATexture3D &ret_tex);
bool plain_text_to_3d_texture(PlainText &plain_text, CUDATexture3D &ret_tex);
bool free_yylvv_resources(YYLVVRes &res);

#endif
