#include "renderstates/arrowglyph.cuh"
#include <vector>
#include <imgui.h>
#include <driver_types.h>
#include "debug_kernels.cuh"
#include "utils.cuh"
#include "app.cuh"

ArrowGlyphRenderState::ArrowGlyphRenderState() : num_arrows_x(100), num_arrows_y(100), vector_length(5.0f),
    arrow_glyph_vao(nullptr), arrow_glyph_program(nullptr), arrow_glyph_graphics_resource(nullptr),
    current_visualizing_z(0.0f), visualize_xy(true) 
{

}

void ArrowGlyphRenderState::initialize(App &app) 
{
    // 4. Evaluate arrow glyphs for z=0, which is a massive vector field of arrows, with VAO structure of (POSITION, COLOR)
    // I actually think z=0 makes more sense (instead of x) so that we can see it first hand when camera is aligned
    initialize_arrow_glyph_resources();
    if (!generate_arrow_glyphs(app, 0.0f)) {
        std::cerr << "Failed to generate arrow glyphs?" << std::endl;
    }
    std::cout << "Use Q and E to switch visualizing plane. Currently visualizing: " << (num_arrows_x * num_arrows_y) << " glyphs." << std::endl
        << "Use P to change up the visualizing plane (XY or YZ)." << std::endl
        << "Use UP ARROW and DOWN ARROW to increase/decrease vector length." << std::endl;
}

void ArrowGlyphRenderState::render(App &app) 
{
    // Show the arrow glyph VAO
    arrow_glyph_program->use();
    glUniformMatrix4fv(arrow_glyph_program->at("model"), 1, GL_FALSE, glm::value_ptr(glm::mat4(1.0f)));
    glUniformMatrix4fv(arrow_glyph_program->at("view"), 1, GL_FALSE, glm::value_ptr(app.camera.view));
    glUniformMatrix4fv(arrow_glyph_program->at("perspective"), 1, GL_FALSE, glm::value_ptr(app.camera.perspective));
    arrow_glyph_vao->draw();
}

void ArrowGlyphRenderState::destroy() 
{
    CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(arrow_glyph_graphics_resource));
    arrow_glyph_graphics_resource = nullptr;
}

void ArrowGlyphRenderState::key_pressed(App &app, int key) 
{
    switch (key) {
        case GLFW_KEY_P:
        {
            visualize_xy = !visualize_xy;
            clamp_visualizing_z(app);
            generate_arrow_glyphs(app, current_visualizing_z);
            break;
        }
    }
}

void ArrowGlyphRenderState::clamp_visualizing_z(App &app) 
{
    if (visualize_xy) 
    {
        current_visualizing_z = glm::clamp(current_visualizing_z, 0.0f, (float) app.res.vf_tex.extent.depth);
    } 
    else 
    {
        current_visualizing_z = glm::clamp(current_visualizing_z, 0.0f, (float) app.res.vf_tex.extent.width);
    }
}

void ArrowGlyphRenderState::process_events(App &app) {
    float layer_change_speed = (visualize_xy ? app.res.vf_tex.extent.depth : app.res.vf_tex.extent.width) * 0.1f;
    
    if (glfwGetKey(app.window, GLFW_KEY_Q)) 
    {
        current_visualizing_z -= layer_change_speed * app.delta_time;
        clamp_visualizing_z(app);
        generate_arrow_glyphs(app, current_visualizing_z);
    }
    
    if (glfwGetKey(app.window, GLFW_KEY_E)) 
    {
        current_visualizing_z += layer_change_speed * app.delta_time;
        clamp_visualizing_z(app);
        generate_arrow_glyphs(app, current_visualizing_z);
    }

    if (glfwGetKey(app.window, GLFW_KEY_UP)) 
    {
        vector_length += 1.0f * app.delta_time;
        generate_arrow_glyphs(app, current_visualizing_z);
    }

    if (glfwGetKey(app.window, GLFW_KEY_DOWN)) 
    {
        vector_length -= 1.0f * app.delta_time;
        vector_length = glm::max(vector_length, 0.1f);
        generate_arrow_glyphs(app, current_visualizing_z);
    }
}

void ArrowGlyphRenderState::initialize_arrow_glyph_resources() 
{
    // arrow glyph vertex memory layout: interleaved
    // PX, PY, PZ, NX, NY, NZ, CR, CG, CB
    int num_arrow_glyphs = num_arrows_x * num_arrows_y;
    int num_vertices = num_arrow_glyphs * 18; // 18 vertices for each arrow - 5 sides with two triangles on one side (4 + 2) * 3
    int num_floats = num_vertices * 9;
    int size_in_bytes = num_floats * sizeof(float);
    std::unique_ptr<float[]> empty_data = std::make_unique<float[]>(num_floats);
    std::memset(empty_data.get(), 0, size_in_bytes);
    arrow_glyph_vao = VAO::make_vao(empty_data.get(),
                                   size_in_bytes,
                                   GL_DYNAMIC_DRAW,
                                   {
                                       VertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 9, nullptr),
                                       VertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 9, (void *) (sizeof(float) * 3)),
                                       VertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 9, (void *) (sizeof(float) * 6))
                                   },
                                   GLDrawCall(GL_TRIANGLES, 0, num_vertices));
    std::cout << "Linking OpenGL VAO with CUDA resource." << std::endl;
    CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&arrow_glyph_graphics_resource, arrow_glyph_vao->vbo, cudaGraphicsMapFlagsNone));
    std::cout << "Compiling arrow glyph visualization shader." << std::endl;
    arrow_glyph_program = Program::make_program("shaders/arrows.vert", "shaders/arrows.frag");
    
    if (!arrow_glyph_program || !arrow_glyph_program->valid) 
    {
        std::cerr << "Failed to compile program?" << std::endl;
    }
}

__device__ inline void set_arrow_glyph(float *float_arr, int offset, const glm::vec3 ag[54]) {
//     float *offset_ptr = &float_arr[offset];
//     std::memcpy(offset_ptr, ag, sizeof(ag));
    for (int i = 0; i < 54; i++) {
        float_arr[offset + (i * 3) + 0] = ag[i].x;
        float_arr[offset + (i * 3) + 1] = ag[i].y;
        float_arr[offset + (i * 3) + 2] = ag[i].z;
    }
}

__global__ void generate_arrow_glyphs_kernel(float *glyphs_vbo_data, CUDATexture3D vf, bool visualize_xy, cudaTextureObject_t vector_magnitude_ctf, float vector_length, float v) 
{
    glm::vec3 arrow[] = 
    {
        // bottom face 1
        glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(1.0f, 0.0f, 1.0f),
        glm::vec3(0.5f, -0.5f, -0.5f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(1.0f, 0.0f, 1.0f),
        glm::vec3(0.5f, -0.5f, 0.5f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(1.0f, 0.0f, 1.0f),

        // bottom face 2
        glm::vec3(0.5f, -0.5f, 0.5f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(1.0f, 0.0f, 1.0f),
        glm::vec3(-0.5f, -0.5f, 0.5f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(1.0f, 0.0f, 1.0f),
        glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(1.0f, 0.0f, 1.0f),

        // front face
        glm::vec3(-0.5f, -0.5f, 0.5f), glm::vec3(0.0f, 0.4472135955f, 0.894427191f), glm::vec3(1.0f, 0.0f, 1.0f),
        glm::vec3(0.5f, -0.5f, 0.5f), glm::vec3(0.0f, 0.4472135955f, 0.894427191f), glm::vec3(1.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, 0.5f, 0.0f), glm::vec3(0.0f, 0.4472135955f, 0.894427191f), glm::vec3(1.0f, 0.0f, 1.0f),

        // back face
        glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(0.0f, 0.4472135955f, -0.894427191f), glm::vec3(1.0f, 0.0f, 1.0f),
        glm::vec3(0.5f, -0.5f, -0.5f), glm::vec3(0.0f, 0.4472135955f, -0.894427191f), glm::vec3(1.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, 0.5f, 0.0f), glm::vec3(0.0f, 0.4472135955f, -0.894427191f), glm::vec3(1.0f, 0.0f, 1.0f),

        // left face
        glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(-0.894427191f, 0.4472135955f, 0.0f), glm::vec3(1.0f, 0.0f, 1.0f),
        glm::vec3(-0.5f, -0.5f, 0.5f), glm::vec3(-0.894427191f, 0.4472135955f, 0.0f), glm::vec3(1.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, 0.5f, 0.0f), glm::vec3(-0.894427191f, 0.4472135955f, 0.0f), glm::vec3(1.0f, 0.0f, 1.0f),

        // right face
        glm::vec3(0.5f, -0.5f, -0.5f), glm::vec3(0.894427191f, 0.4472135955f, 0.0f), glm::vec3(1.0f, 0.0f, 1.0f),
        glm::vec3(0.5f, -0.5f, 0.5f), glm::vec3(0.894427191f, 0.4472135955f, 0.0f), glm::vec3(1.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, 0.5f, 0.0f), glm::vec3(0.894427191f, 0.4472135955f, 0.0f), glm::vec3(1.0f, 0.0f, 1.0f)
    };

    glm::vec3 ori(0.0f, 0.0f, 0.0f);
    float4 vector = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float max_vec_length = 0.0f;
    int index = (blockIdx.y * gridDim.x + blockIdx.x) * 162; // pitch of 162 floats (POS, NOR, COLOR) * 18
    
    if (visualize_xy) 
    {
        int x = blockIdx.x;
        int y = blockIdx.y;
        float sx = ((float) x / gridDim.x) * vf.extent.width;
        float sy = ((float) y / gridDim.y) * vf.extent.height;
        float sz = v;
        max_vec_length = glm::min(1.0f / gridDim.x * vf.extent.width, 1.0f / gridDim.y * vf.extent.height);
        vector = tex3D<float4>(vf.texture, sx, sy, sz);
        ori = glm::vec3(sx, sy, sz);
    } 
    else 
    {
        float y = blockIdx.x;
        float z = blockIdx.y;
        float sx = v;
        float sy = ((float) y / gridDim.x) * vf.extent.height;
        float sz = ((float) z / gridDim.y) * vf.extent.depth;
        max_vec_length = glm::min(1.0f / gridDim.x * vf.extent.height, 1.0f / gridDim.y * vf.extent.depth);
        vector = tex3D<float4>(vf.texture, sx, sy, sz);
        ori = glm::vec3(sx, sy, sz);
    }

    float3 normalized = float3_normalize(vector);
    glm::vec3 up(normalized.x, normalized.y, normalized.z);
    glm::vec3 random_front = glm::vec3(0.0f, 0.0f, 1.0f);
    glm::vec3 front = glm::normalize(glm::cross(random_front, up));
    glm::vec3 right = glm::normalize(glm::cross(up, front));
    glm::mat3 ruf = glm::mat3(right, up, front);
    float len = max_vec_length * (vector.w / vf.longest_vector) * vector_length;
    float4 color = tex1D<float4>(vector_magnitude_ctf, vector.w / vf.longest_vector * 4.0f + 0.5f);

    for (int i = 0; i < 18; i++) 
    {
        arrow[i * 3].y += 0.5f;
        arrow[i * 3].y *= len;
        arrow[i * 3] = ruf * arrow[i * 3];
        arrow[i * 3] += ori;
        arrow[i * 3 + 1] = ruf * arrow[i * 3 + 1];
        arrow[i * 3 + 2] = glm::vec3(color.x, color.y, color.z);
    }

    set_arrow_glyph(glyphs_vbo_data, index, arrow);
}

bool ArrowGlyphRenderState::generate_arrow_glyphs(App &app, float z) 
{
    int num_arrow_glyphs = num_arrows_x * num_arrows_y;
    int num_vertices = num_arrow_glyphs * 18;
    int num_floats = num_vertices * 9;
    int size_in_bytes = num_floats * sizeof(float);
    float *glyphs_vbo_data = nullptr; // mom, I am on GPU!
    size_t size_in_bytes_mapped;
    CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &arrow_glyph_graphics_resource));
    CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer((void **) &glyphs_vbo_data, &size_in_bytes_mapped, arrow_glyph_graphics_resource));
    assert(size_in_bytes_mapped == size_in_bytes);
    dim3 num_blocks(num_arrows_x, num_arrows_y, 1);
    // num_threads can be 1 for all I care because between-thread communication is not required
    generate_arrow_glyphs_kernel<<<num_blocks, 1>>>(glyphs_vbo_data, app.res.vf_tex, visualize_xy, app.ctf_tex_cuda, vector_length, z);
//     while (true) {
//         int idx;
//         std::cin >> idx;
//         if (idx < 0) {
//             break;
//         }
//         float3 res = access_data_on_device<float3>((float3 *) glyphs_vbo_data, idx);
//         std::cout << res.x << ", " << res.y << ", " << res.z << std::endl;
//     }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &arrow_glyph_graphics_resource));
    return true;
}

void ArrowGlyphRenderState::draw_user_controls(App &app)
{
    ImGui::SetNextWindowPos({220.0f, 0.0f}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({app.screen_width - 220.0f, 140}, ImGuiCond_FirstUseEver);
    
    bool should_update = false;

    if (ImGui::Begin("Arrow Glyph Controls"))
    {
        float cap = (visualize_xy ? app.res.vf_tex.extent.depth : app.res.vf_tex.extent.width);

        should_update |= ImGui::SliderFloat("Vector length", &vector_length, 1.0f, 20.0f);
        should_update |= ImGui::SliderFloat("Visualizing plane", &current_visualizing_z, 0.0f, cap);

        ImGui::Text("Visualizing plane");
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::BeginListBox("plane"))
        {
            if (ImGui::Selectable("X-Y plane", visualize_xy))
            {
                visualize_xy = true;
                should_update = true;
                clamp_visualizing_z(app);
            }
            if (ImGui::Selectable("Y-Z plane", !visualize_xy))
            {
                visualize_xy = false;
                should_update = true;
                clamp_visualizing_z(app);
            }
            ImGui::EndListBox();
        }
    }

    if (should_update)
    {
        generate_arrow_glyphs(app, current_visualizing_z);
    }

    ImGui::End();
}
