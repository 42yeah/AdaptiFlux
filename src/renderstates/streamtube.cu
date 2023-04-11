#include "renderstates/streamtube.cuh"
#include <fstream>
#include <random>
#include <imgui.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "app.cuh"

struct StreamLineVertex
{
    glm::vec3 position;
    glm::vec3 color;
};

struct StreamTubeVertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 color;
};


StreamTubeRenderState::StreamTubeRenderState() : StreamLineRenderState()
{
    streamtube_radius = 1.0f;
    shadow_mapping = false;
    streamtube_graphics_resource = nullptr;
}

StreamTubeRenderState::~StreamTubeRenderState()
{
    StreamLineRenderState::~StreamLineRenderState();
}

void StreamTubeRenderState::initialize(App &app)
{
    StreamLineRenderState::initialize(app);

    if (!allocate_graphics_resources())
    {
        std::cerr << "Failed to allocate streamtube graphics resources?" << std::endl;        
    }

    generate_streamtubes();
}

bool StreamTubeRenderState::allocate_graphics_resources()
{
    int num_streamline_vertices = num_seeds * num_lines * 2;
    // 18 vertices per 2 control points
    int num_streamtube_vertices = (num_streamline_vertices) / 2 * 9;
    // (vertex, normal, color)
    int num_floats = num_streamtube_vertices * 9;
    int size_in_bytes = sizeof(float) * num_floats;

    std::cout << "Allocating " << size_in_bytes << " bytes (" << (size_in_bytes / 1024) << "K) for streamtube rendering." << std::endl;

    std::unique_ptr<float[]> empty_data = std::make_unique<float[]>(num_floats);
    std::memset(empty_data.get(), 0, size_in_bytes);
    streamtube_vao = std::make_shared<VAO>(empty_data.get(), size_in_bytes, GL_DYNAMIC_DRAW, 
        std::vector<VertexAttribPointer>(
            {
                VertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 9, nullptr),
                VertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 9, (void *) (sizeof(float) * 3)),
                VertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 9, (void *) (sizeof(float) * 6))
            }
        ),
        GLDrawCall(GL_TRIANGLES, 0, num_streamtube_vertices));

    std::cout << "# streamtube vertices: " << num_streamtube_vertices << std::endl;

    CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&streamtube_graphics_resource, streamtube_vao->vbo, cudaGraphicsMapFlagsNone));
    
    test_streamtube_generation();

    streamtube_program = Program::make_program("shaders/streamtube.vert", "shaders/streamtube.frag");
    if (streamtube_program == nullptr || !streamtube_program->valid)
    {
        std::cerr << "Invalid streamtube program?" << std::endl;
        return false;
    }
    return true;
}

struct ExportFace
{
    int a, b, c;
};

bool StreamTubeRenderState::export_streamtube_vbo_as_obj(const std::string &path)
{
    glBindVertexArray(streamtube_vao->vao);
    glBindBuffer(GL_ARRAY_BUFFER, streamtube_vao->vbo);
    StreamTubeVertex *verts = (StreamTubeVertex *) (glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY));
    int num_verts = streamtube_vao->draw_call.size;

    std::ofstream output(path);
    if (!output.good())
    {
        std::cerr << "Bad output: " << path << "?" << std::endl;
    }

    // Since there are 200 num_seeds, there should be 200 tubes
    int current_vertex_id = 1;
    std::vector<StreamTubeVertex> vertices;
    std::vector<ExportFace> faces;
    for (int i = 0; i < num_seeds; i++)
    {
        int base_vertices_offset = i * (num_lines * 9);
        
        int j = 0;
        while (true)
        {
            StreamTubeVertex v1 = verts[base_vertices_offset + j + 0];
            StreamTubeVertex v2 = verts[base_vertices_offset + j + 1];
            StreamTubeVertex v3 = verts[base_vertices_offset + j + 2];
            int num_zero_verts = 0.0f;
            num_zero_verts += v1.position == glm::vec3(0.0f) ? 1 : 0;
            num_zero_verts += v2.position == glm::vec3(0.0f) ? 1 : 0;
            num_zero_verts += v3.position == glm::vec3(0.0f) ? 1 : 0;
            assert(num_zero_verts == 3 || num_zero_verts == 0); // An incomplete vertex means this assert will fail
            if (num_zero_verts == 3)
            {
                break;
            }
            vertices.insert(vertices.end(), {v1, v2, v3});
            faces.insert(faces.end(), {current_vertex_id, current_vertex_id + 1, current_vertex_id + 2});
            current_vertex_id += 3;

            output << "v " << v1.position.x << " " << v1.position.y << " " << v1.position.z << std::endl;
            output << "v " << v2.position.x << " " << v2.position.y << " " << v2.position.z << std::endl;
            output << "v " << v3.position.x << " " << v3.position.y << " " << v3.position.z << std::endl;
            j += 3;
        }
    }

    for (int i = 0; i < vertices.size(); i += 3)
    {
        output << "vt 0.0 0.0" << std::endl;
        output << "vt 0.0 0.0" << std::endl;
        output << "vt 0.0 0.0" << std::endl;
    }

    for (int i = 0; i < vertices.size(); i += 3)
    {
        output << "vn " << vertices[i].normal.x << " " << vertices[i].normal.y << " " << vertices[i].normal.z << std::endl;
        output << "vn " << vertices[i + 1].normal.x << " " << vertices[i + 1].normal.y << " " << vertices[i + 1].normal.z << std::endl;
        output << "vn " << vertices[i + 2].normal.x << " " << vertices[i + 2].normal.y << " " << vertices[i + 2].normal.z << std::endl;
    }

    for (const ExportFace &f : faces)
    {
        output << "f " << f.a << "/" << f.a << "/" << f.a << " " << f.b << "/" << f.b << "/" << f.b << " " << f.c << "/" << f.c << "/" << f.c << std::endl;
    }
    output.close();

    glUnmapBuffer(GL_ARRAY_BUFFER);
    return true;
}

void StreamTubeRenderState::destroy()
{
    StreamLineRenderState::destroy();

    CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(streamtube_graphics_resource));
    streamtube_graphics_resource = nullptr;
}

void StreamTubeRenderState::render(App &app)
{
    streamtube_program->use();
    glUniformMatrix4fv(streamtube_program->at("model"), 1, GL_FALSE, glm::value_ptr(glm::mat4(1.0f)));
    glUniformMatrix4fv(streamtube_program->at("view"), 1, GL_FALSE, glm::value_ptr(app.camera.view));
    glUniformMatrix4fv(streamtube_program->at("perspective"), 1, GL_FALSE, glm::value_ptr(app.camera.perspective));
    streamtube_vao->draw();

    if (render_seed_points)
    {
        glPointSize(point_size);
        seed_points_program->use();
        glUniformMatrix4fv(seed_points_program->at("model"), 1, GL_FALSE, glm::value_ptr(glm::mat4(1.0f)));
        glUniformMatrix4fv(seed_points_program->at("view"), 1, GL_FALSE, glm::value_ptr(app.camera.view));
        glUniformMatrix4fv(seed_points_program->at("perspective"), 1, GL_FALSE, glm::value_ptr(app.camera.perspective));
        seed_points_vao->draw();
    }
}

void StreamTubeRenderState::process_events(App &app)
{
    StreamLineRenderState::process_events(app);
}

void StreamTubeRenderState::key_pressed(App &app, int key)
{
    StreamLineRenderState::key_pressed(app, key);
    switch (key)
    {
        case GLFW_KEY_R:
        case GLFW_KEY_T:
        case GLFW_KEY_O:
        case GLFW_KEY_P:
        case GLFW_KEY_LEFT_BRACKET:
        case GLFW_KEY_RIGHT_BRACKET:
        case GLFW_KEY_PERIOD:
            generate_streamtubes();
            break;
    }
}

void StreamTubeRenderState::draw_user_controls(App &app)
{    
    ImGui::SetNextWindowPos({220.0f, 0.0f}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({app.screen_width - 220.0f, 140}, ImGuiCond_FirstUseEver);
    
    bool should_update = false;

    if (ImGui::Begin("Streamtube Controls"))
    {
        should_update |= ImGui::SliderFloat("Simulation delta time", &simulation_dt, 0.001f, 1.0f);
        if (ImGui::Button("Reset"))
        {
            simulation_dt = 1.0f / 256.0f;
            should_update = true;
        }
        if (ImGui::RadioButton("Delta wing recommended strategy", seed_points_strategy == 0)) { seed_points_strategy = 0; should_update = true; }
        if (ImGui::RadioButton("Line", seed_points_strategy == 1)) { seed_points_strategy = 1; should_update = true; }
        if (ImGui::RadioButton("Rect", seed_points_strategy == 2)) { seed_points_strategy = 2; should_update = true; }
        if (seed_points_strategy != 0 && ImGui::CollapsingHeader("Seeding strategy"))
        {
            ImGui::Text("Bounding box: (%f %f %f)", app.delta_wing_bounding_box.max.x,
                app.delta_wing_bounding_box.max.y,
                app.delta_wing_bounding_box.max.z);

            should_update |= ImGui::InputFloat3("Seed begin", (float *) &seed_begin);
            should_update |= ImGui::InputFloat3("Seed end", (float *) &seed_end);
            ImGui::Text("Seeding plane offset axis");
        }
        if (seed_points_strategy == 0)
        {
            should_update |= ImGui::SliderFloat("Seeding plane (X axis)", &seeding_plane_x, 0.0f, app.res.vf_tex.extent.width);
            if (ImGui::Button("Go to critical region"))
            {
                seeding_plane_x = 51.0f;
                should_update = true;
            }
        }
        should_update |= ImGui::Checkbox("Use Runge-Kutta 4 integrator", &use_runge_kutta_4_integrator);
        should_update |= ImGui::Checkbox("Adaptive seeding", &adaptive_mode);

        if (adaptive_mode)
        {
            if (ImGui::CollapsingHeader("Adaptive mode properties"))
            {
                // should_update |= ImGui::SliderFloat("Seed point generation threshold", &seed_point_threshold, 0.001f, app.res.vf_tex.longest_vector);
                should_update |= ImGui::SliderFloat("Seed point generation threshold", &seed_point_threshold, 0.001f, 1.0f);
                should_update |= ImGui::SliderFloat("Adaptive explosion radius", &adaptive_explosion_radius, 1.0f, 20.0f);
                should_update |= ImGui::SliderInt("Number of explosions", &num_explosion, 1, 10);
                should_update |= ImGui::SliderInt("Explosion cooldown counter", &explosion_cooldown_counter, 1, 4000);
            }
        }

        should_update |= ImGui::Checkbox("Streamtube simplification", &do_simplify);
        if (do_simplify)
        {
            if (ImGui::CollapsingHeader("Simplification properties"))
            {
                should_update |= ImGui::SliderFloat("Simplification threshold", &distortion_threshold, 1.001f, 1.5f);
            }
        }

        ImGui::Checkbox("Render seed points", &render_seed_points);
        if (render_seed_points)
        {
            ImGui::SliderFloat("Seed point point size", &point_size, 1.0f, 20.0f);
        }
        
        should_update |= ImGui::SliderFloat("Streamtube radius", &streamtube_radius, 0.1f, 10.0f);
        should_update |= ImGui::Checkbox("Enable shadow mapping", &shadow_mapping);

        ImGui::End();

        if (should_update)
        {
            generate_streamlines(app);
            generate_streamtubes();
        }
    }
}


//
// Stores information about the creation of streamtube.
//
struct StreamTubeInfo
{
    int streamline_starting_index;
    int streamtube_starting_index;
    int streamline_index;
    int streamtube_index;
};

__global__ void streamtube_kernel(float *streamtube_vbo_data,
                                  size_t streamtube_stride,
                                  float *streamline_vbo_data,
                                  size_t streamline_stride,
                                  int num_seeds,
                                  float streamtube_radius,
                                  StreamTubeInfo *info)
{
    int seed_index = blockIdx.y * gridDim.x + blockIdx.x;
    if (seed_index >= num_seeds)
    {
        return;
    }

    int streamline_starting_index = seed_index * streamline_stride;
    int streamtube_starting_index = seed_index * streamtube_stride;
    int streamline_index = streamline_starting_index;
    int streamtube_index = streamtube_starting_index;

    memset(&streamtube_vbo_data[streamtube_starting_index], 0, sizeof(float) * streamtube_stride);

    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    unsigned int order[18] = 
    {
        0, 1, 4,
        0, 4, 3,
        0, 2, 5,
        0, 5, 3,
        1, 2, 4,
        4, 2, 5
    };
    while (true)
    {
        StreamLineVertex &streamline_vert_a = (*(StreamLineVertex *) &(streamline_vbo_data[streamline_index]));
        StreamLineVertex &streamline_vert_b = (*(StreamLineVertex *) &(streamline_vbo_data[streamline_index + 6]));

        if (streamline_vert_a.position == glm::vec3(0.0f) || streamline_vert_b.position == glm::vec3(0.0f))
        {
            break;
        }

        glm::vec3 front = glm::normalize(streamline_vert_b.position - streamline_vert_a.position);
        // TODO: cross operation might take a long time.
        glm::vec3 right = glm::normalize(glm::cross(front, up));
        glm::vec3 up = glm::normalize(glm::cross(right, front));

        // Three (3) new vertices spawns from each. A empty triangular prism will be formed.
        StreamTubeVertex tube_vertices[6];
        for (int i = 0; i < 3; i++)
        {
            float rot = ((float) (i + 1) / 3) * 2.0f * glm::pi<float>();
            glm::vec3 tube_left_pos = streamline_vert_a.position + streamtube_radius * (right * cosf(rot) + up * sinf(rot));
            glm::vec3 tube_right_pos = streamline_vert_b.position + streamtube_radius * (right * cosf(rot) + up * sinf(rot));
            StreamTubeVertex left;
            left.position = tube_left_pos;
            left.normal = glm::vec3(0.0f); // TODO: TBD
            left.color = streamline_vert_a.color;
            StreamTubeVertex right;
            right.position = tube_right_pos;
            right.normal = glm::vec3(0.0f);
            right.color = streamline_vert_b.color;
            tube_vertices[i] = left;
            tube_vertices[i + 3] = right;
        }
        
        StreamTubeVertex *indices = (StreamTubeVertex *) &(streamtube_vbo_data[streamtube_index]);
        for (int i = 0; i < 18; i++)
        {
            indices[i] = tube_vertices[order[i]];
        }

        streamline_index += 2 * 6;
        streamtube_index += 18 * 9;
    }

    info[seed_index].streamline_starting_index = streamline_starting_index;
    info[seed_index].streamline_index = streamline_index;
    info[seed_index].streamtube_starting_index = streamtube_starting_index;
    info[seed_index].streamtube_index = streamtube_index;

}

bool StreamTubeRenderState::generate_streamtubes()
{
    int num_blocks_x = 32;
    int num_blocks_y = (num_seeds + num_blocks_x - 1) / num_blocks_x;
    
    std::cout << "Streamtube generation report: block count: " << num_blocks_x << "x" << num_blocks_y << std::endl;
    dim3 num_blocks(num_blocks_x, num_blocks_y, 1);

    // Map streamline and streamtube data.
    float *streamline_vbo_data;
    size_t mapped_size;
    size_t streamline_stride = num_lines * 2 * 6;
    size_t expected = num_seeds * streamline_stride * sizeof(float);
    CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &streamline_graphics_resource));
    CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer((void **) (&streamline_vbo_data), &mapped_size, streamline_graphics_resource));
    std::cout << "Mapped size: " << mapped_size << " bytes as opposed to the expected of " << expected << std::endl;

    float *streamtube_vbo_data;
    size_t streamtube_stride = num_lines * 9 * 9;
    CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &streamtube_graphics_resource));
    CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer((void **) (&streamtube_vbo_data), &mapped_size, streamtube_graphics_resource));
    expected = num_seeds * streamtube_stride * sizeof(float);
    std::cout << "Mapped size: " << mapped_size << " bytes as opposed to the expected of " << expected << std::endl;

    std::unique_ptr<StreamTubeInfo[]> info = std::make_unique<StreamTubeInfo[]>(num_blocks.x * num_blocks.y);
    StreamTubeInfo *info_cuda = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&info_cuda, num_blocks.x * num_blocks.y * sizeof(StreamTubeInfo)));
    CHECK_CUDA_ERROR(cudaMemcpy(info_cuda, info.get(), num_blocks.x * num_blocks.y * sizeof(StreamTubeInfo), cudaMemcpyHostToDevice));

    streamtube_kernel<<<num_blocks, 1>>>(streamtube_vbo_data, streamtube_stride, streamline_vbo_data, streamline_stride, num_seeds, streamtube_radius, info_cuda);

    CHECK_CUDA_ERROR(cudaMemcpy(info.get(), info_cuda, num_blocks.x * num_blocks.y * sizeof(StreamTubeInfo), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaFree(info_cuda));

    // for (int i = 0; i < num_blocks.y; i++)
    // {
    //     for (int j = 0; j < num_blocks.x; j++)
    //     {
    //         int idx = i * num_blocks.x + j;

    //         std::cout << "BLK " << idx << ": SLSI " << info[idx].streamline_starting_index << " (" << (info[idx].streamline_starting_index / 6) << "); " << 
    //             "SLI " << info[idx].streamline_index << " (" << (info[idx].streamline_index / 6) << "); " <<
    //             "STSI " << info[idx].streamtube_starting_index << " (" << (info[idx].streamtube_starting_index / 9) << "); " <<
    //             "STI " << info[idx].streamtube_index << " (" << (info[idx].streamtube_index / 9) << ")" << std::endl; 
    //     }
    // }

    CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &streamline_graphics_resource));
    CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &streamtube_graphics_resource));
    return true;
}

bool StreamTubeRenderState::generate_bare_bones_streamline()
{
    glBindVertexArray(streamline_vao->vao);
    glBindBuffer(GL_ARRAY_BUFFER, streamline_vao->vbo);
    StreamLineVertex *verts = (StreamLineVertex *) (glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
    size_t verts_size_in_bytes = sizeof(StreamLineVertex) * num_seeds * num_lines * 2;
    std::memset(verts, 0, verts_size_in_bytes);
    std::cout << "Clearing up " << verts_size_in_bytes << " bytes (" << (verts_size_in_bytes / sizeof(float)) << " floats)" << std::endl;

    StreamLineVertex replacements[6];
    
    std::uniform_real_distribution<float> distrib;
    std::random_device dev;

    for (int i = 0; i < 6; i++)
    {
        replacements[i].position = glm::vec3(distrib(dev) * 100.0f, distrib(dev) * 100.0f, distrib(dev) * 100.0f);
        replacements[i].color = glm::vec3(distrib(dev), distrib(dev), distrib(dev));
    }
    for (int i = 0; i < 5; i++)
    {
        verts[i * 2 + 0] = replacements[i];
        verts[i * 2 + 1] = replacements[i + 1];
    }

    glUnmapBuffer(GL_ARRAY_BUFFER);
    return true;
}

bool StreamTubeRenderState::test_streamtube_generation()
{
    if (!generate_bare_bones_streamline())
    {
        std::cerr << "Cannot generate bare bones streamline." << std::endl;
        return false;
    }
    if (!generate_streamtubes())
    {
        std::cerr << "Cannot generate streamtubes." << std::endl;
        return false;
    }
    if (!export_streamtube_vbo_as_obj("test.obj"))
    {
        std::cerr << "Failed to export streamtube obj." << std::endl;
        return false;
    }
    return true;
}
