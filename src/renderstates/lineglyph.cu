#include "renderstates/lineglyph.cuh"
#include <vector>
#include <imgui.h>
#include "debug_kernels.cuh"
#include "utils.cuh"
#include "app.cuh"

LineGlyphRenderState::LineGlyphRenderState() : num_lines_x(200), num_lines_y(200), vector_length(3.0f),
    line_glyph_vao(nullptr), line_glyph_program(nullptr), line_glyph_graphics_resource(nullptr),
    current_visualizing_z(0.0f), visualize_xy(true) 
{

}

void LineGlyphRenderState::initialize(App &app) 
{
    // 4. Evaluate line glyphs for z=0, which is a massive vector field of lines, with VAO structure of (POSITION, COLOR)
    // I actually think z=0 makes more sense (instead of x) so that we can see it first hand when camera is aligned
    initialize_line_glyph_resources();
    
    if (!generate_line_glyphs(app, 0.0f)) 
    {
        std::cerr << "Failed to generate line glyphs?" << std::endl;
    }
    
    std::cout << "Use Q and E to switch visualizing plane. Currently visualizing: " << (num_lines_x * num_lines_y) << " glyphs." << std::endl
        << "Use P to change up the visualizing plane (XY or YZ)." << std::endl;
}

void LineGlyphRenderState::render(App &app) 
{
    // Show the line glyph VAO
    line_glyph_program->use();
    glUniformMatrix4fv(line_glyph_program->at("model"), 1, GL_FALSE, glm::value_ptr(glm::mat4(1.0f)));
    glUniformMatrix4fv(line_glyph_program->at("view"), 1, GL_FALSE, glm::value_ptr(app.camera.view));
    glUniformMatrix4fv(line_glyph_program->at("perspective"), 1, GL_FALSE, glm::value_ptr(app.camera.perspective));
    line_glyph_vao->draw();
}

void LineGlyphRenderState::destroy() 
{
    CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(line_glyph_graphics_resource));
    line_glyph_graphics_resource = nullptr;
}

void LineGlyphRenderState::key_pressed(App &app, int key) 
{
    switch (key) 
    {
        case GLFW_KEY_P:
        {
            visualize_xy = !visualize_xy;
            clamp_visualizing_z(app);
            generate_line_glyphs(app, current_visualizing_z);
            break;
        }
    }
}

void LineGlyphRenderState::clamp_visualizing_z(App &app) 
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

void LineGlyphRenderState::process_events(App &app) 
{
    float layer_change_speed = (visualize_xy ? app.res.vf_tex.extent.depth : app.res.vf_tex.extent.width) * 0.1f;
    
    if (glfwGetKey(app.window, GLFW_KEY_Q)) 
    {
        current_visualizing_z -= layer_change_speed * app.delta_time;
        clamp_visualizing_z(app);
        generate_line_glyphs(app, current_visualizing_z);
    }

    if (glfwGetKey(app.window, GLFW_KEY_E)) 
    {
        current_visualizing_z += layer_change_speed * app.delta_time;
        clamp_visualizing_z(app);
        generate_line_glyphs(app, current_visualizing_z);
    }
}

void LineGlyphRenderState::initialize_line_glyph_resources() 
{
    // line glyph vertex memory layout: interleaved
    // PX, PY, PZ, CR, CG, CB
    int num_line_glyphs = num_lines_x * num_lines_y;
    int num_vertices = num_line_glyphs * 2;
    int num_floats = num_vertices * 6;
    int size_in_bytes = num_floats * sizeof(float);
    std::unique_ptr<float[]> empty_data = std::make_unique<float[]>(num_floats);
    std::memset(empty_data.get(), 0, size_in_bytes);
    line_glyph_vao = VAO::make_vao(empty_data.get(),
                                   size_in_bytes,
                                   GL_DYNAMIC_DRAW,
                                   {
                                       VertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, nullptr),
                                       VertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, (void *) (sizeof(float) * 3))
                                   },
                                   GLDrawCall(GL_LINES, 0, num_vertices));
    std::cout << "Linking OpenGL VAO with CUDA resource." << std::endl;
    CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&line_glyph_graphics_resource, line_glyph_vao->vbo, cudaGraphicsMapFlagsNone));
    std::cout << "Compiling line glyph visualization shader." << std::endl;
    line_glyph_program = Program::make_program("shaders/vectors.vert", "shaders/vectors.frag");
    
    if (!line_glyph_program || !line_glyph_program->valid) 
    {
        std::cerr << "Failed to compile program?" << std::endl;
    }
}

__global__ void generate_line_glyphs_kernel(float *glyphs_vbo_data, CUDATexture3D vf, bool visualize_xy, cudaTextureObject_t vector_magnitude_ctf, float vector_length, float v) 
{
    float3 ori = make_float3(0.0f, 0.0f, 0.0f);
    float4 vector = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float max_vec_length = 0.0f;
    int index = (blockIdx.y * gridDim.x + blockIdx.x) * 12; // pitch of 12 floats (POS, COLOR) * 2
    
    if (visualize_xy) 
    {
        int x = blockIdx.x;
        int y = blockIdx.y;
        float sx = ((float) x / gridDim.x) * vf.extent.width;
        float sy = ((float) y / gridDim.y) * vf.extent.height;
        float sz = v;
        max_vec_length = glm::min(1.0f / gridDim.x * vf.extent.width, 1.0f / gridDim.y * vf.extent.height);
        vector = tex3D<float4>(vf.texture, sx, sy, sz);
        ori = make_float3(sx, sy, sz);
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
        ori = make_float3(sx, sy, sz);
    }
    
    float3 tar = float3_add(ori, float3_scale(max_vec_length * (vector.w / vf.longest_vector) * vector_length, float3_normalize(vector)));
    float4 color = tex1D<float4>(vector_magnitude_ctf, vector.w / vf.longest_vector * 4.0f + 0.5f);
    set_float3(glyphs_vbo_data, index, ori);
    set_float3(glyphs_vbo_data, index + 3, make_float3(color.x, color.y, color.z));
    set_float3(glyphs_vbo_data, index + 6, tar);
    set_float3(glyphs_vbo_data, index + 9, make_float3(color.x, color.y, color.z));
    // set_float3(glyphs_vbo_data, 0, make_float3(max_vec_length, vf.longest_vector, 0.0f));
    // set_float3(glyphs_vbo_data, 3, make_float3(blockDim.x, blockDim.y, 0.0f));
}

bool LineGlyphRenderState::generate_line_glyphs(App &app, float z) 
{
    int num_line_glyphs = num_lines_x * num_lines_y;
    int num_vertices = num_line_glyphs * 2;
    int num_floats = num_vertices * 6;
    int size_in_bytes = num_floats * sizeof(float);
    float *glyphs_vbo_data = nullptr; // mom, I am on GPU!
    size_t size_in_bytes_mapped;
    CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &line_glyph_graphics_resource));
    CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer((void **) &glyphs_vbo_data, &size_in_bytes_mapped, line_glyph_graphics_resource));
    assert(size_in_bytes_mapped == size_in_bytes);
    dim3 num_blocks(num_lines_x, num_lines_y, 1);
    // num_threads can be 1 for all I care because between-thread communication is not required
    generate_line_glyphs_kernel<<<num_blocks, 1>>>(glyphs_vbo_data, app.res.vf_tex, visualize_xy, app.ctf_tex_cuda, vector_length, z);
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
    CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &line_glyph_graphics_resource));
    return true;
}

void LineGlyphRenderState::draw_user_controls(App &app)
{
    ImGui::SetNextWindowPos({220.0f, 0.0f}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({app.screen_width - 220.0f, 140}, ImGuiCond_FirstUseEver);
    
    bool should_update = false || app.framerate_history.stress_test;

    if (ImGui::Begin("Line Glyph Controls"))
    {
        float cap = (visualize_xy ? app.res.vf_tex.extent.depth : app.res.vf_tex.extent.width);

        should_update |= ImGui::SliderFloat("Vector length", &vector_length, 1.0f, 5.0f);
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
        generate_line_glyphs(app, current_visualizing_z);
    }

    ImGui::End();
}
