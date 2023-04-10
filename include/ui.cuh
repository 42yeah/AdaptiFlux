#ifndef YYLVV_UI_CUH
#define YYLVV_UI_CUH

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "yylvv.cuh"
#include "VAO.h"
#include "Program.h"
#include "Camera.h"
#include <VectorField.h> // for bounding box
#include "renderstate.cuh"
#include "renderstates/lineglyph.cuh"
#include "renderstates/arrowglyph.cuh"
#include "renderstates/streamline.cuh"

struct UIRes
{
    std::shared_ptr<VAO> bounding_box_vao; // VAO for bounding box
    std::shared_ptr<Program> bounding_box_program; // program for line-drawing (without color mapping)
    Camera camera; // camera for seeing things
    std::shared_ptr<RenderState> render_state; // current RenderState - supports line glyphs, etc.
    int screen_width, screen_height;
    double last_instant; // time-related variable
    float delta_time; // elapsed time from last frame

    // delta wing related stuffs
    BBox delta_wing_bounding_box;
    std::shared_ptr<VAO> delta_wing_vao;
    std::shared_ptr<Program> delta_wing_program;

    // color transfer function texture
    cudaArray_t ctf_data_cuda;
    cudaTextureObject_t ctf_tex_cuda;
};


void print_controls();
bool initialize_ui_resources(GLFWwindow *window, UIRes &res);
bool initialize_delta_wing_resources(YYLVVRes &res, UIRes &ui_res);
bool initialize_color_transfer_function(UIRes &ui_res);
void switch_state(YYLVVRes &res, UIRes &ui_res, std::shared_ptr<RenderState> new_state);
void align_camera(UIRes &ui_res);
void start_ui(YYLVVRes &res);
void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods);
void cursor_pos_callback(GLFWwindow *window, double xpos, double ypos);
bool cleanup_ui_resources(UIRes &res);
void draw_delta_wing(UIRes &ui_res);


#ifdef YYLVV_UI_IMPL

UIRes *bound_ui_res = nullptr;
YYLVVRes *bound_yylvv_res = nullptr;

void print_controls()
{
    std::cout << "Use camera to navigate around." << std::endl
            << "Activate line glyph view by pressing button L." << std::endl
            << "Activate arrow glyph view by pressing button G." << std::endl
            << "Activate streamline view by pressing button Z." << std::endl;
}

void switch_state(YYLVVRes &res, UIRes &ui_res, std::shared_ptr<RenderState> new_state)
{
    if (ui_res.render_state != nullptr)
    {
        ui_res.render_state->destroy();
    }
    ui_res.render_state = new_state;
    ui_res.render_state->initialize(res, ui_res);
}

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (action != 1)
    {
        return;
    }
    
    UIRes &ui_res = *bound_ui_res;
    YYLVVRes &res = *bound_yylvv_res;
    switch (key)
    {
        case GLFW_KEY_L:
            switch_state(res, ui_res, std::make_shared<LineGlyphRenderState>());
            break;

        case GLFW_KEY_G:
            switch_state(res, ui_res, std::make_shared<ArrowGlyphRenderState>());
            break;

        case GLFW_KEY_Z:
            switch_state(res, ui_res, std::make_shared<StreamLineRenderState>());
            break;
    }

    if (ui_res.render_state)
    {
        ui_res.render_state->key_pressed(res, ui_res, key);
    }
}

void cursor_pos_callback(GLFWwindow *window, double xpos, double ypos)
{
    Camera &camera = bound_ui_res->camera;

    // ypos is (0, 0) at the bottom left corner and (w, h) at top right corner, thus it needs to be flipped
    // since right is actually our left (help me handedness,) we need to flip xpos as well
    ypos = -ypos;
    xpos = -xpos;
    if (!camera.prev_cursor_pos)
    {
        camera.prev_cursor_pos = glm::dvec2(xpos, ypos);
        return;
    }
    glm::dvec2 curr_pos = glm::dvec2(xpos, ypos);
    glm::dvec2 delta_pos = curr_pos - *camera.prev_cursor_pos;
    camera.yaw += delta_pos.x * camera.sensitivity;
    camera.pitch += delta_pos.y * camera.sensitivity;
    camera.prev_cursor_pos = curr_pos;
    camera.update_components(bound_ui_res->screen_width, bound_ui_res->screen_height);
}

bool initialize_ui_resources(GLFWwindow *window, UIRes &res)
{
    std::cout << "Initializing bounding box and bounding box program." << std::endl;
    const glm::vec3 a = glm::vec3(-0.5f, -0.5f, -0.5f),
                b = glm::vec3(0.5f, -0.5f, -0.5f),
                c = glm::vec3(0.5f, -0.5f, 0.5f),
                d = glm::vec3(-0.5f, -0.5f, 0.5f),
                e = a + glm::vec3(0.0f, 1.0f, 0.0f),
                f = b + glm::vec3(0.0f, 1.0f, 0.0f),
                g = c + glm::vec3(0.0f, 1.0f, 0.0f),
                h = d + glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 bounding_box_data[] = {
            a, b, b, c, c, d, d, a,
            e, f, f, g, g, h, h, e,
            a, b, b, f, f, e, e, a,
            d, c, c, g, g, h, h, d,
            a, d, d, h, h, e, e, a,
            b, c, c, g, g, f, f, b
    };
    res.bounding_box_vao = VAO::make_vao(bounding_box_data,
                                         sizeof(bounding_box_data),
                                         GL_STATIC_DRAW,
                                         {VertexAttribPointer(0,
                                                              3,
                                                              GL_FLOAT,
                                                              GL_FALSE,
                                                              sizeof(float) * 3,
                                                              nullptr)},
                                         GLDrawCall(GL_LINES, 0, 48));
    std::cout << "Compiling line drawing program." << std::endl;
    res.bounding_box_program = Program::make_program("shaders/lines.vert", "shaders/lines.frag");
    if (!res.bounding_box_program || !res.bounding_box_program->valid) {
        std::cerr << "Cannot link line-drawing program?" << std::endl;
        return false;
    }
    std::cout << "Configuring OpenGL & GLFW." << std::endl;
    glEnable(GL_DEPTH_TEST);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    res.render_state = nullptr;
    glfwGetFramebufferSize(window, &res.screen_width, &res.screen_height);
    glfwSwapInterval(1);
    res.last_instant = glfwGetTime();
    res.delta_time = 0.0f;
    return true;
}

bool initialize_delta_wing_resources(YYLVVRes &res, UIRes &ui_res)
{
    std::cout << "Evaluating delta wing bounding box and allocating graphics resources." << std::endl;
    ui_res.delta_wing_bounding_box = res.vf_tex.get_bounding_box();
    float x_min = 50.0f;
    float x_max = 169.0f;
    float y_min = 27.5f;
    float y_mid = 100.0f;
    float y_max = 172.5f;
    float z = 0.0f;
    std::vector<float3> delta_wing_fl3 = {
        make_float3(x_min, y_mid, z),
        make_float3(x_max, y_min, z),
        make_float3(x_max, y_max, z)
    };
    std::cout << "Compiling delta wing shader." << std::endl;
    ui_res.delta_wing_vao = VAO::make_vao(delta_wing_fl3.data(), delta_wing_fl3.size() * sizeof(float3), GL_STATIC_DRAW,
                                   {
                                       VertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, nullptr)
                                   },
                                   GLDrawCall(GL_TRIANGLES, 0, 3));
    ui_res.delta_wing_program = Program::make_program("shaders/lines.vert", "shaders/delta.frag");
    if (!ui_res.delta_wing_program || !ui_res.delta_wing_program->valid)
    {
        return false;
    }
    return true;
}

bool initialize_color_transfer_function(UIRes &ui_res)
{
    std::vector<float4> vector_magnitude_ctf;
    std::cout << "Initializing color transfer function for CUDA: creating device array." << std::endl;
    vector_magnitude_ctf.push_back(make_float4(0.4f, 0.6f, 0.9f, 1.0f)); // TODO: 1D texture is weird because it has a
    vector_magnitude_ctf.push_back(make_float4(0.0f, 1.0f, 0.0f, 1.0f)); //       weird padding of 0.5
    vector_magnitude_ctf.push_back(make_float4(0.9f, 0.9f, 0.0f, 1.0f));
    vector_magnitude_ctf.push_back(make_float4(1.0f, 0.0f, 0.0f, 1.0f));
    int vm_size_in_bytes = vector_magnitude_ctf.size() * sizeof(float4);
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
    CHECK_CUDA_ERROR(cudaMallocArray(&ui_res.ctf_data_cuda, &desc, vector_magnitude_ctf.size(), 0, 0));
    CHECK_CUDA_ERROR(cudaMemcpy2DToArray(ui_res.ctf_data_cuda, 0, 0, vector_magnitude_ctf.data(), vm_size_in_bytes, vm_size_in_bytes, 1, cudaMemcpyHostToDevice));

    std::cout << "Creating vector magnitude CTF texture." << std::endl;
    cudaResourceDesc rdesc;
    std::memset(&rdesc, 0, sizeof(cudaResourceDesc));
    {
        rdesc.resType = cudaResourceTypeArray;
        rdesc.res.array.array = ui_res.ctf_data_cuda;
    }
    cudaTextureDesc tdesc;
    std::memset(&tdesc, 0, sizeof(cudaTextureDesc));
    {
        tdesc.addressMode[0] = cudaAddressModeWrap;
        tdesc.filterMode = cudaFilterModeLinear;
        tdesc.readMode = cudaReadModeElementType;
        tdesc.normalizedCoords = 0; // let's try normalizing it
    }
    CHECK_CUDA_ERROR(cudaCreateTextureObject(&ui_res.ctf_tex_cuda, &rdesc, &tdesc, nullptr));
    std::cout << "Vector magnitude CTF texture creation complete." << std::endl;

    return true;
}

void handle_continuous_key_events(YYLVVRes &res, UIRes &ui_res)
{
    Camera &camera = ui_res.camera;
    if (glfwGetKey(res.window, GLFW_KEY_W))
    {
        camera.eye += camera.front * camera.speed * ui_res.delta_time;
        camera.update_components(ui_res.screen_width, ui_res.screen_height);
    }
    if (glfwGetKey(res.window, GLFW_KEY_S))
    {
        camera.eye -= camera.front * camera.speed * ui_res.delta_time;
        camera.update_components(ui_res.screen_width, ui_res.screen_height);
    }
    if (glfwGetKey(res.window, GLFW_KEY_A))
    {
        camera.eye -= camera.right * camera.speed * ui_res.delta_time;
        camera.update_components(ui_res.screen_width, ui_res.screen_height);
    }
    if (glfwGetKey(res.window, GLFW_KEY_D))
    {
        camera.eye += camera.right * camera.speed * ui_res.delta_time;
        camera.update_components(ui_res.screen_width, ui_res.screen_height);
    }
}

void update_delta_time(UIRes &res)
{
    double this_instant = glfwGetTime();
    res.delta_time = (float) (this_instant - res.last_instant);
    res.last_instant = this_instant;
}

void align_camera(UIRes &ui_res)
{
    glm::vec3 extent = ui_res.delta_wing_bounding_box.extend(); // TODO: a typo
    float max_ext = glm::max(glm::max(extent.x, extent.y), extent.z);
    float init_dist = glm::max(extent.x, extent.y) * 0.5f * sqrt(3.0f);
    ui_res.camera.eye = ui_res.delta_wing_bounding_box.center() - glm::vec3(0.0f, 0.0f, init_dist);
    ui_res.camera.speed = max_ext * 0.1f; // whole thing in 10 seconds
    ui_res.camera.z_near = 1.0f;
    ui_res.camera.z_far = max_ext * 2.0f + fabs(init_dist);
    ui_res.camera.update_components(ui_res.screen_width, ui_res.screen_height);
}

void draw_delta_wing(UIRes &ui_res)
{
    // 1. Draw the bounding box (that we calculated)
    ui_res.bounding_box_program->use();
    glm::mat4 model = glm::translate(glm::mat4(1.0f), ui_res.delta_wing_bounding_box.center());
    model = glm::scale(model, ui_res.delta_wing_bounding_box.extend());
    glUniformMatrix4fv(ui_res.bounding_box_program->at("model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(ui_res.bounding_box_program->at("view"), 1, GL_FALSE, glm::value_ptr(ui_res.camera.view));
    glUniformMatrix4fv(ui_res.bounding_box_program->at("perspective"), 1, GL_FALSE, glm::value_ptr(ui_res.camera.perspective));
    ui_res.bounding_box_vao->draw();

    // 2. Draw the delta wing triangle
    ui_res.delta_wing_program->use();
    glUniformMatrix4fv(ui_res.delta_wing_program->at("model"), 1, GL_FALSE, glm::value_ptr(glm::mat4(1.0f)));
    glUniformMatrix4fv(ui_res.delta_wing_program->at("view"), 1, GL_FALSE, glm::value_ptr(ui_res.camera.view));
    glUniformMatrix4fv(ui_res.delta_wing_program->at("perspective"), 1, GL_FALSE, glm::value_ptr(ui_res.camera.perspective));
    ui_res.delta_wing_vao->draw();
}

void start_ui(YYLVVRes &res)
{
    GLFWwindow *window = res.window;
    print_controls();
    UIRes ui_res;
    if (!initialize_ui_resources(window, ui_res))
    {
        std::cerr << "Things might not be rendered correctly: failed to initialize UI resources?" << std::endl;
    }
    if (!initialize_delta_wing_resources(res, ui_res))
    {
        std::cerr << "The delta wing might not be rendered: cannot initialize delta wing resources?" << std::endl;
    }
    if (!initialize_color_transfer_function(ui_res))
    {
        std::cerr << "Cannot initialize color transfer function?" << std::endl;
    }
    std::cout << "Aligning camera." << std::endl;
    align_camera(ui_res);
    while (!glfwWindowShouldClose(window))
    {
        bound_ui_res = &ui_res;
        bound_yylvv_res = &res;
        glfwPollEvents();
        update_delta_time(ui_res);
        handle_continuous_key_events(res, ui_res);
        if (ui_res.render_state)
        {
            ui_res.render_state->process_events(res, ui_res);
        }
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        draw_delta_wing(ui_res);
        if (ui_res.render_state)
        {
            ui_res.render_state->render(res, ui_res);
        }
        glfwSwapBuffers(window);
    }
    if (!cleanup_ui_resources(ui_res))
    {
        std::cerr << "Cannot cleanup UI resources?" << std::endl;
    }
}

bool cleanup_ui_resources(UIRes &res)
{
    std::cout << "Cleaning up UI resources..." << std::endl;
    if (res.render_state)
    {
        res.render_state->destroy();
    }
    CHECK_CUDA_ERROR(cudaDestroyTextureObject(res.ctf_tex_cuda));
    CHECK_CUDA_ERROR(cudaFreeArray(res.ctf_data_cuda));
    return true;
}

#endif
#endif

