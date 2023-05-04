#include "renderstates/streamline.cuh"
#include <iostream>
#include "app.cuh"
#include "utils.cuh"
#include "debug_kernels.cuh"
#include <imgui.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <chrono>

StreamLineRenderState::StreamLineRenderState() : num_seeds(200),
        num_lines(8192),
        simulation_dt(1.0f / 256.0f),
        streamline_vao(nullptr),
        streamline_program(nullptr),
        streamline_graphics_resource(nullptr),
        use_runge_kutta_4_integrator(false),
        seeding_plane_x(0.0f),
        adaptive_mode(false),
        adaptive_explosion_radius(1.0f),
        num_explosion(3),
        explosion_cooldown_counter(100),
        seed_point_threshold(0.5f),
        do_simplify(false),
        distortion_threshold(1.01f),
        seed_points_strategy(0),
        seed_begin(0.0f),
        seed_end(0.0f),
        render_seed_points(false),
        seed_points_vao(nullptr),
        seed_points_program(nullptr),
        point_size(3.0f)
{

}

StreamLineRenderState::~StreamLineRenderState() 
{

}

void StreamLineRenderState::initialize(App &app) {
    std::cout << "Use 'R' to toggle runge-kutta 4 integrator." << std::endl;
    std::cout << "Use 'T' to toggle adaptive mode." << std::endl;
    std::cout << "Adaptive mode controls:" << std::endl
        << "    O and P to control the seed explosion count." << std::endl
        << "    [ and ] to control the explosion cooldown." << std::endl
        << "    = and - to control the explosion radius." << std::endl;
    std::cout << "Use '.' to toggle streamline simplification." << std::endl;

    // seed_point_threshold = app.res.vf_tex.longest_vector * 0.5f;

    if (!allocate_graphics_resources()) {
        std::cerr << "Faield to allocate graphics resources?" << std::endl;
    }
    if (!generate_streamlines(app)) {
        std::cerr << "Failed to generate streamlines?" << std::endl;
    }
}

bool StreamLineRenderState::allocate_graphics_resources() {
    std::cout << "Allocating streamline graphics resources." << std::endl;

    // streamline memory view: (POS, COLOR); 6 floats; 24 bytes
    // there will be num_seeds * num_lines streamlines, num_lines means how many line segments PER streamline
    int num_vertices = (num_seeds * num_lines) * 2;
    int num_floats = num_vertices * 6;
    int size_in_bytes = num_floats * sizeof(float);
    
    std::cout << "Allocating " << size_in_bytes << " bytes (" << (size_in_bytes / 1024) << "K) for streamline." << std::endl;

    std::unique_ptr<float[]> empty_data = std::make_unique<float[]>(num_floats);
    streamline_vao = VAO::make_vao(empty_data.get(), size_in_bytes, GL_DYNAMIC_DRAW,
                                   {
                                       VertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, nullptr),
                                       VertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, (void *) (sizeof(float) * 3))
                                   },
                                   GLDrawCall(GL_LINES, 0, num_vertices));
    std::cout << "Linking OpenGL VAO with CUDA resource." << std::endl;
    CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&streamline_graphics_resource, streamline_vao->vbo, cudaGraphicsMapFlagsNone));
    streamline_program = Program::make_program("shaders/vectors.vert", "shaders/vectors.frag");
    
    if (!streamline_program || !streamline_program->valid) 
    {
        std::cerr << "Invalid streamline program?" << std::endl;
        return false;
    }

    seed_points_program = Program::make_program("shaders/seedpoints.vert", "shaders/seedpoints.frag");

    if (!seed_points_program || !seed_points_program->valid) 
    {
        std::cerr << "Invalid seed point program?" << std::endl;
        return false;
    }

    empty_data.reset(new float[3 * num_seeds]);
    seed_points_vao = VAO::make_vao(empty_data.get(), sizeof(glm::vec3) * num_seeds, GL_DYNAMIC_DRAW,
                                   {
                                       VertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, nullptr)
                                   },
                                   GLDrawCall(GL_POINTS, 0, num_seeds));

    return true;
}

void StreamLineRenderState::destroy() 
{
    std::cout << "Streamline is being destroyed. Good night." << std::endl;
    CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(streamline_graphics_resource));
    streamline_graphics_resource = nullptr;
}

void StreamLineRenderState::render(App &app) 
{
    streamline_program->use();
    glUniformMatrix4fv(streamline_program->at("model"), 1, GL_FALSE, glm::value_ptr(glm::mat4(1.0f)));
    glUniformMatrix4fv(streamline_program->at("view"), 1, GL_FALSE, glm::value_ptr(app.camera.view));
    glUniformMatrix4fv(streamline_program->at("perspective"), 1, GL_FALSE, glm::value_ptr(app.camera.perspective));
    streamline_vao->draw();

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

void StreamLineRenderState::process_events(App &app) 
{
    if (glfwGetKey(app.window, GLFW_KEY_UP))
    {
        seeding_plane_x += 10.0f * app.delta_time;
        generate_streamlines(app);
    }
    if (glfwGetKey(app.window, GLFW_KEY_DOWN))
    {
        seeding_plane_x -= 10.0f * app.delta_time;
        generate_streamlines(app);
    }
    if (glfwGetKey(app.window, GLFW_KEY_MINUS))
    {
        adaptive_explosion_radius -= 1.0f * app.delta_time;
        std::cout << "Current explosion radius: " << adaptive_explosion_radius << std::endl;
        generate_streamlines(app);
    }
    if (glfwGetKey(app.window, GLFW_KEY_EQUAL))
    {
        adaptive_explosion_radius += 1.0f * app.delta_time;
        std::cout << "Current explosion radius: " << adaptive_explosion_radius << std::endl;
        generate_streamlines(app);
    }
}

void StreamLineRenderState::key_pressed(App &app, int key) 
{
    switch (key) 
    {
        case GLFW_KEY_R:
        {
            use_runge_kutta_4_integrator = !use_runge_kutta_4_integrator;
            if (!generate_streamlines(app))
            {
                std::cerr << "Failed to generate streamlines?" << std::endl;
            }
            break;
        }

        case GLFW_KEY_T:
        {
            adaptive_mode = !adaptive_mode;
            if (!generate_streamlines(app)) 
            {
                std::cerr << "Failed to generate streamlines?" << std::endl;
            }
            break;
        }

        case GLFW_KEY_O:
        {
            num_explosion--;
            std::cout << "Seed explosion count: " << num_explosion << std::endl;
            generate_streamlines(app);
            break;
        }

        case GLFW_KEY_P:
        {
            num_explosion++;
            std::cout << "Seed explosion count: " << num_explosion << std::endl;
            generate_streamlines(app);
            break;
        }

        case GLFW_KEY_LEFT_BRACKET:
        {
            explosion_cooldown_counter--;
            std::cout << "Explosion cooldown counter: " << explosion_cooldown_counter << std::endl;
            generate_streamlines(app);
            break;
        }

        case GLFW_KEY_RIGHT_BRACKET:
        {
            explosion_cooldown_counter++;
            std::cout << "Explosion cooldown counter: " << explosion_cooldown_counter << std::endl;
            generate_streamlines(app);
            break;
        }

        case GLFW_KEY_PERIOD:
        {
            do_simplify = !do_simplify;
            if (do_simplify)
            {
                std::cout << "Streamline simplification is now ON." << std::endl;
            }
            else
            {
                std::cout << "Streamline simplification is now OFF." << std::endl;
            }
            generate_streamlines(app);
            break;
        }
    }
}

// generate seeds across y plane because that's what the original blog post does
// TODO: add other ways to generate seed
bool StreamLineRenderState::generate_seed_points_delta_wing(const BBox &bbox, int num_seeds) 
{
    seed_points.clear();
    glm::vec3 step = glm::vec3(0.0f,
                               (bbox.max - bbox.min).y / (num_seeds - 1),
                               0.0f);
    glm::vec3 current_seed = bbox.min + glm::vec3(0.1f, 0.1f, 0.1f);
    current_seed.x += seeding_plane_x;
    for (int i = 0; i < num_seeds; i++) 
    {
        seed_points.push_back(current_seed);
        current_seed += step;
    }
    return true;
}

bool StreamLineRenderState::generate_seed_points_line(glm::vec3 a, glm::vec3 b, int num_seeds)
{
    seed_points.clear();
    glm::vec3 step = (b - a) * ((1.0f) / (num_seeds - 1));
    glm::vec3 current_seed = a;

    for (int i = 0; i < num_seeds; i++)
    {
        seed_points.push_back(current_seed);
        current_seed += step;
    }
    return true;
}

bool StreamLineRenderState::generate_seed_points_rect(glm::vec3 a, glm::vec3 b, int num_seeds)
{
    seed_points.clear();
    int width = (int) sqrtf(num_seeds);
    int height = num_seeds / width;
    glm::vec3 xy = b - a;
    xy.z = 0.0f;

    glm::vec3 xy_steps = xy * ((1.0f) / (width - 1));
    float z_steps = (b - a).z / (height - 1);

    glm::vec3 current_seed = a;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            seed_points.push_back(current_seed);
            current_seed += xy_steps;
        }
        current_seed = glm::vec3(a.x, a.y, a.z + z_steps * y);
    }
    std::cout << "Rectangular seed generation: " << seed_points.size() << " seeds generated." << std::endl;
    return true;
}

bool StreamLineRenderState::generate_seed_points(App &app, int num_seeds)
{
    if (seed_points_strategy == 0)
    {
        return generate_seed_points_delta_wing(app.delta_wing_bounding_box, num_seeds);
    }
    else if (seed_points_strategy == 1)
    {
        return generate_seed_points_line(seed_begin, seed_end, num_seeds);
    }
    else
    {
        return generate_seed_points_rect(seed_begin, seed_end, num_seeds);
    }
}

bool StreamLineRenderState::generate_streamlines(App &app) 
{
    const auto started = std::chrono::high_resolution_clock::now();
    if (adaptive_mode)
    {
        if (!trace_streamlines_adaptive(app))
        {
            std::cerr << "Failed to adaptively trace streamlines?" << std::endl;
        }
    }
    else // just generate streamlines as normal
    {
        // glm::vec3 begin_seeds = glm::vec3(0.1f, 10.1f, seeding_plane_x);
        // glm::vec3 end_seeds = begin_seeds;
        // end_seeds.x = app.delta_wing_bounding_box.max.x;
        if (!generate_seed_points(app, num_seeds))
        {
            std::cerr << "Failed to generate seed points?" << std::endl;
            return false;
        }
        if (!trace_streamlines(app)) 
        {
            std::cerr << "Failed to trace streamlines?" << std::endl;
            return false;
        }
    }
    const auto streamlines_traced = std::chrono::high_resolution_clock::now();
    std::cout << "Profile: streamlines traced: " << ((streamlines_traced - started).count() * 1e-6) << "ms" << std::endl;
    if (!finalize_seed_points(app))
    {
        std::cerr << "Failed to finalize seed points?" << std::endl;
    }
    const auto seedpoints_finalized = std::chrono::high_resolution_clock::now();
    if (do_simplify)
    {
        if (!simplify_streamlines())
        {
            std::cerr << "Failed to simplify streamlines?" << std::endl;
        }
        const auto streamlines_simplified = std::chrono::high_resolution_clock::now();
        std::cout << "Profile: streamlines simplified: " << ((streamlines_simplified - seedpoints_finalized).count() * 1e-6) << "ms" << std::endl;
    }
    const auto done = std::chrono::high_resolution_clock::now();
    std::cout << "Total: " << ((done - started).count() * 1e-6) << "ms" << std::endl;

    return true;
}

__device__ float4 euler_integrate(glm::vec3 p, float dt, CUDATexture3D vf) 
{
    float4 v = tex3D<float4>(vf.texture, p.x, p.y, p.z);
    return make_float4(v.x * dt, v.y * dt, v.z * dt, v.w);
}

__device__ float4 runge_kutta_4_integrate(glm::vec3 p, float dt, CUDATexture3D vf) 
{
    float4 k1 = tex3D<float4>(vf.texture, p.x, p.y, p.z);
    const float dt_half = 0.5f * dt;
    float4 k2 = tex3D<float4>(vf.texture, p.x + dt_half * k1.x, p.y + dt_half * k1.y, p.z + dt_half * k1.z);
    float4 k3 = tex3D<float4>(vf.texture, p.x + dt_half * k2.x, p.y + dt_half * k2.y, p.z + dt_half * k2.z);
    float4 k4 = tex3D<float4>(vf.texture, p.x + dt * k3.x, p.y + dt * k3.y, p.z + dt * k3.z);

    constexpr float one_over_six = 0.166666f;
    return make_float4
    (
        one_over_six * dt * (k1.x + 2.0f * k2.x + 2.0f * k3.x + k4.x),
        one_over_six * dt * (k1.y + 2.0f * k2.y + 2.0f * k3.y + k4.y),
        one_over_six * dt * (k1.z + 2.0f * k2.z + 2.0f * k3.z + k4.z),
        one_over_six * (k1.w + 2.0f * k2.w + 2.0f * k3.w + k4.w)
    );
}

__device__ bool is_inside_bounding_box(const BBox &bbox, const glm::vec3 &p) 
{
    return (p.x > bbox.min.x && p.y > bbox.min.y && p.z > bbox.min.z && p.x < bbox.max.x && p.y < bbox.max.y && p.z < bbox.max.z);
}

__global__ void trace_streamlines_kernel(glm::vec3 *seed_points, int num_seeds, int num_lines, float *streamline_vbo_data, CUDATexture3D vf, float dt, cudaTextureObject_t vector_magnitude_ctf, BBox vf_bbox, bool use_rk4) 
{
    // streamline data: {POSITION, COLOR}
    int index = blockIdx.y * gridDim.x + blockIdx.x;
    int streamline_stride = num_lines * 2 * 6;

    while (index < num_seeds) 
    {
        // seed point is located at seed_points[index]
        glm::vec3 sp = seed_points[index];
        int line_start = index * streamline_stride;

        for (int i = 0; i < num_lines - 1; i++) 
        {
            if (!is_inside_bounding_box(vf_bbox, sp)) 
            {
                set_vec3(streamline_vbo_data, line_start + i * 12, glm::vec3(0.0f));
                set_vec3(streamline_vbo_data, line_start + i * 12 + 6, glm::vec3(0.0f));
                continue;
            }

            float4 dv = use_rk4 ? runge_kutta_4_integrate(sp, dt, vf) : euler_integrate(sp, dt, vf);
            float4 color = tex1D<float4>(vector_magnitude_ctf, dv.w / vf.longest_vector * 4.0f + 0.5f);

            set_vec3(streamline_vbo_data, line_start + i * 12, sp);
            set_xyz(streamline_vbo_data, line_start + i * 12 + 3, color.x, color.y, color.z);
            
            sp.x += dv.x;
            sp.y += dv.y;
            sp.z += dv.z;

            set_vec3(streamline_vbo_data, line_start + i * 12 + 6, sp);
            set_xyz(streamline_vbo_data, line_start + i * 12 + 9, color.x, color.y, color.z);
        }

        index += gridDim.x * gridDim.y;
    }
}

bool StreamLineRenderState::trace_streamlines(App &app) 
{
    int num_blocks_x = 32;
    int num_blocks_y = (seed_points.size() + (num_blocks_x - 1)) / num_blocks_x;
    std::cout << "Trace report: parallel blocks: (" << num_blocks_x << ", " << num_blocks_y << ")" << std::endl;
    dim3 num_blocks(num_blocks_x, num_blocks_y, 1);

    float *vbo_data;
    size_t mapped_size;
    CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &streamline_graphics_resource));
    CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer((void **) &vbo_data, &mapped_size, streamline_graphics_resource));

    glm::vec3 *seed_points_cuda;
    size_t seed_points_size = sizeof(glm::vec3) * seed_points.size();
    CHECK_CUDA_ERROR(cudaMalloc(&seed_points_cuda, seed_points_size));
    CHECK_CUDA_ERROR(cudaMemcpy(seed_points_cuda, seed_points.data(), seed_points_size, cudaMemcpyHostToDevice));

    // assert(seed_points.size() == num_seeds);
    trace_streamlines_kernel<<<num_blocks, 1>>>(seed_points_cuda, num_seeds, num_lines, vbo_data, app.res.vf_tex, simulation_dt, app.ctf_tex_cuda, app.delta_wing_bounding_box, use_runge_kutta_4_integrator);

    // while (true)
    // {
    //     int index;
    //     std::cin >> index;
    //     glm::vec3 fl = access_data_on_device<glm::vec3>((glm::vec3 *) vbo_data, index * 3);
    //     std::cout << fl.x << ", " << fl.y << ", " << fl.z << std::endl;
    // }

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &streamline_graphics_resource));
    CHECK_CUDA_ERROR(cudaFree(seed_points_cuda));

    return true;
}

__device__ glm::vec3 evaluate_vector_delta(const glm::vec3 &p, CUDATexture3D vf)
{
    constexpr float epsilon = 0.01f;
    constexpr float one_over_eps = 1.0f / epsilon;
    
    float4 v = tex3D<float4>(vf.texture, p.x, p.y, p.z);
    float4 vx = tex3D<float4>(vf.texture, p.x - epsilon, p.y, p.z);
    float4 vy = tex3D<float4>(vf.texture, p.x, p.y - epsilon, p.z);
    float4 vz = tex3D<float4>(vf.texture, p.x, p.y, p.z - epsilon);

    return one_over_eps * glm::vec3(v.x - vx.x, v.y - vy.y, v.z - vz.z);
}

__device__ glm::vec3 evaluate_vector_slope(const glm::vec3 &p, CUDATexture3D vf, int axis)
{
    constexpr float epsilon = 0.01f;
    constexpr float one_over_eps = 1.0f / epsilon;

    float4 v = tex3D<float4>(vf.texture, p.x, p.y, p.z);
    float4 vv;

    switch (axis)
    {
        case 0:
            vv = tex3D<float4>(vf.texture, p.x - epsilon, p.y, p.z);
            break;

        case 1:
            vv = tex3D<float4>(vf.texture, p.x, p.y - epsilon, p.z);
            break;

        case 2:
            vv = tex3D<float4>(vf.texture, p.x, p.y, p.z - epsilon);
            break;
    }

    return one_over_eps * glm::vec3(v.x - vv.x, v.y - vv.y, v.z - vv.z);
}

__device__ float evaluate_jacobian_det(const glm::vec3 &p, CUDATexture3D vf)
{
    glm::mat3 m(evaluate_vector_slope(p, vf, 0), 
        evaluate_vector_slope(p, vf, 1), 
        evaluate_vector_slope(p, vf, 2));
    return glm::determinant(m);
}

__device__ glm::vec3 evaluate_vector_delta_delta(const glm::vec3 &p, CUDATexture3D vf)
{
    constexpr float epsilon = 0.01f;
    constexpr float one_over_eps = 1.0f / epsilon;

    glm::vec3 dv = evaluate_vector_delta(p, vf);
    glm::vec3 dvx = evaluate_vector_delta(p - glm::vec3(epsilon, 0.0f, 0.0f), vf);
    glm::vec3 dvy = evaluate_vector_delta(p - glm::vec3(0.0f, epsilon, 0.0f), vf);
    glm::vec3 dvz = evaluate_vector_delta(p - glm::vec3(0.0f, 0.0f, epsilon), vf);
    
    return one_over_eps * glm::vec3(dv.x - dvx.x, dv.y - dvy.y, dv.z - dvz.z);
}

__device__ float curvature(const glm::vec3 &p, CUDATexture3D vf)
{
    glm::vec3 dv = evaluate_vector_delta(p, vf);
    glm::vec3 ddv = evaluate_vector_delta_delta(p, vf);
    return glm::length(glm::cross(dv, ddv)) / powf(glm::length(dv), 3);
}


struct TraceInfo
{
    int index;
    float maximum_metric;
    float average_metric;
    float curand;
    int streamline_size;
    float streamline_length;
    float streamline_dist;
    float global_distortion;
};

__device__ bool jacobian_metric(const glm::vec3 &sp, const CUDATexture3D &vf, float threshold)
{
    return evaluate_jacobian_det(sp, vf) > threshold;
}

// I think this doesn't make sense
__device__ bool length_metric(const glm::vec3 &sp, const CUDATexture3D &vf, float threshold)
{
    return glm::length(evaluate_vector_delta(sp, vf)) > threshold;
}

__device__ bool critical_metric(const glm::vec3 &sp, const CUDATexture3D &vf, float threshold, glm::vec3 &vdir, glm::vec3 &right, glm::vec3 &up)
{
    float4 v = tex3D<float4>(vf.texture, sp.x, sp.y, sp.z);

    // Step 1: determine if this is a dead zone
    if (v.w == 0.0f)
    {
        return false;
    }

    // Step 2: determine if vector magnitude is way too small (which means both x, y, and z are minimal)
    constexpr float infinitesimal_scale = 0.1f;
    if (v.w < infinitesimal_scale * vf.average_vector)
    {
        return true;
    }

    // Step 3: determine if vectors around it are too poorly aligned
    vdir = glm::normalize(glm::vec3(v.x, v.y, v.z));
    float test_dot = fabs(glm::dot(vdir, glm::vec3(0.0f, 1.0f, 0.0f)));
    glm::vec3 fake_up = glm::vec3(0.0f, 1.0f, 0.0f);
    if (test_dot > 0.95f)
    {
        fake_up = glm::vec3(0.0f, 0.0f, 1.0f);
    }
    right = glm::normalize(glm::cross(vdir, fake_up));
    up = glm::normalize(glm::cross(right, vdir));

    constexpr float sample_scale = 1.0f;
    float a = fabs(glm::dot(vdir, glm::normalize(float4_to_vec3(sample_texture(vf.texture, sp + sample_scale * v.w * right)))));
    float b = fabs(glm::dot(vdir, glm::normalize(float4_to_vec3(sample_texture(vf.texture, sp + sample_scale * v.w * up)))));
    float c = fabs(glm::dot(vdir, glm::normalize(float4_to_vec3(sample_texture(vf.texture, sp - sample_scale * v.w * right)))));
    float d = fabs(glm::dot(vdir, glm::normalize(float4_to_vec3(sample_texture(vf.texture, sp - sample_scale * v.w * up)))));

    // Step 4: Check if they have poor alignment
    return ((a + b + c + d) * 0.25f < threshold);
}

__device__ glm::vec3 seed_uniform_sphere(const glm::vec3 &sp, float explosion_radius, curandState *curand_state)
{
    float theta = 2.0f * glm::pi<float>() * curand_uniform(curand_state);
    float phi = acosf(1.0f - 2.0f * curand_uniform(curand_state));
    glm::vec3 offset_dir(sinf(phi) * cosf(theta), sinf(phi) * sinf(theta), cosf(phi));
    return sp + offset_dir * explosion_radius;
}

__device__ glm::vec3 random_direction(curandState *curand_state)
{
    float t = 2.0f * glm::pi<float>() * curand_uniform(curand_state);
    float p = acosf(1.0f - 2.0f * curand_uniform(curand_state));
    return glm::vec3(sinf(p) * cosf(t), sinf(p) * sinf(t), cosf(p));
}

__device__ glm::vec3 evaluate_vector_slope_across_plane(const glm::vec3 &p, const glm::vec3 &right, const glm::vec3 &up, CUDATexture3D vf, int axis)
{
    constexpr float epsilon = 0.01f;
    constexpr float one_over_eps = 1.0f / epsilon;

    float4 v = tex3D<float4>(vf.texture, p.x, p.y, p.z);
    float4 vv;
    glm::vec3 p0 = p;

    switch (axis)
    {
        case 0:
            p0 -= right * epsilon;
            vv = tex3D<float4>(vf.texture, p0.x, p0.y, p0.z);
            break;

        case 1:
            p0 -= up * epsilon;
            vv = tex3D<float4>(vf.texture, p0.x, p0.y, p0.z);
            break;
    }

    return one_over_eps * glm::vec3(v.x - vv.x, v.y - vv.y, v.z - vv.z);
}

__device__ glm::vec3 seed_flow_guided(const glm::vec3 &sp, 
                                      float explosion_radius, 
                                      curandState *curand_state, 
                                      const glm::vec3 &vdir, 
                                      const glm::vec3 &right, 
                                      const glm::vec3 &up,
                                      CUDATexture3D vf,
                                      const glm::vec3 &rand_dir,
                                      int i,
                                      int n)
{
    float theta = acosf(vdir.y) + 0.001f;

    float phi;
    if (vdir.x >= 0.0f)
    {
        phi = acosf(glm::clamp(vdir.z / sinf(theta), -1.0f, 1.0f));
    }
    else
    {
        phi = acosf(glm::clamp(-vdir.z / sinf(theta), -1.0f, 1.0f)) + glm::pi<float>();
    }

    glm::mat3 rotate(cosf(phi), cosf(theta) * sinf(phi), sinf(theta) * sinf(phi),
        0.0f, -sinf(theta), cosf(theta),
        -sinf(phi), cosf(theta) * cosf(phi), sinf(theta) * cosf(phi));
    rotate = glm::transpose(rotate);

    // Align gradients with vector field vector
    glm::vec3 a = rotate * evaluate_vector_slope_across_plane(sp, right, up, vf, 0);
    glm::vec3 b = rotate * evaluate_vector_slope_across_plane(sp, right, up, vf, 1);

    // 3b1b's quick trick for computing eigenvalues
    float m = (a.x + b.y) / 2.0f;
    float p = a.x * b.y - a.y * b.x;

    // If there is no imaginary part, we seed it in a circular manner; 
    // otherwise we assume it's a center, and seed it in a straight line (in a random direction.)
    float s = (float) i / n;
    if (m * m - p >= 0.0f)
    {
        s *= 2.0f * glm::pi<float>();
        return sp + explosion_radius * (cosf(s) * right + sinf(s) * up);
    }
    else
    {
        return sp + explosion_radius * (float) (i + 1) * rand_dir;
    }
}

__global__ void trace_and_generate_kernel(glm::vec3 *seed_points, 
                                          int *num_current_seeds,
                                          int num_maximum_seeds, 
                                          int num_lines,
                                          int trace_start,
                                          int trace_end,
                                          float *streamline_vbo_data, 
                                          CUDATexture3D vf, 
                                          float dt, 
                                          cudaTextureObject_t vector_magnitude_ctf, 
                                          BBox vf_bbox, 
                                          bool use_rk4,
                                          float threshold,
                                          float explosion_radius,
                                          int num_explosion,
                                          int explosion_cooldown_counter,
                                          int initial_explosion_cooldown,
                                          TraceInfo *debug)
{
    int index = blockIdx.y * gridDim.x + blockIdx.x + trace_start;
    int streamline_stride = num_lines * 2 * 6;

    curandState curand_state;
    curand_init(1337, index, 0, &curand_state);

    while (index <= trace_end) 
    {
        // seed point is located at seed_points[index]
        glm::vec3 sp = seed_points[index];
        int line_start = index * streamline_stride;

        int explosion_cooldown = initial_explosion_cooldown;

        // Debug info
        float maximum_metric = 0.0f;
        float average_metric = 0.0f;

        for (int i = 0; i < num_lines - 1; i++) 
        {
            if (!is_inside_bounding_box(vf_bbox, sp)) 
            {
                set_vec3(streamline_vbo_data, line_start + i * 12, glm::vec3(0.0f));
                set_vec3(streamline_vbo_data, line_start + i * 12 + 6, glm::vec3(0.0f));
                continue;
            }

            float4 dv = use_rk4 ? runge_kutta_4_integrate(sp, dt, vf) : euler_integrate(sp, dt, vf);
            float4 color = tex1D<float4>(vector_magnitude_ctf, dv.w / vf.longest_vector * 4.0f + 0.5f);

            set_vec3(streamline_vbo_data, line_start + i * 12, sp);
            set_xyz(streamline_vbo_data, line_start + i * 12 + 3, color.x, color.y, color.z);
            
            sp.x += dv.x;
            sp.y += dv.y;
            sp.z += dv.z;

            set_vec3(streamline_vbo_data, line_start + i * 12 + 6, sp);
            set_xyz(streamline_vbo_data, line_start + i * 12 + 9, color.x, color.y, color.z);
 
            // float metric = jacobian_metric(sp, vf);
            // maximum_metric = glm::max(metric, maximum_metric);
            // average_metric += metric;
            // bool is_critical = jacobian_metric(sp, vf, threshold);

            // Vector direction
            glm::vec3 vdir, right, up;
            bool is_critical = critical_metric(sp, vf, threshold, vdir, right, up);

            if (is_critical && explosion_cooldown <= 0)
            {
                // Generate a seed point directly above the current sample point
                int new_seed_point_index = atomicAdd(num_current_seeds, num_explosion);
                explosion_cooldown = explosion_cooldown_counter;

                if (new_seed_point_index >= num_maximum_seeds)
                {
                    atomicAdd(num_current_seeds, -num_explosion);
                    continue;
                }

                int num_new_seeds = glm::min(num_maximum_seeds - new_seed_point_index, num_explosion);
                glm::vec3 rdir = random_direction(&curand_state);
                for (int j = 0; j < num_new_seeds; j++)
                {
                    // seed_points[new_seed_point_index + j] = seed_uniform_sphere(sp, explosion_radius, &curand_state);
                    seed_points[new_seed_point_index + j] = seed_flow_guided(sp, 
                        explosion_radius, 
                        &curand_state, 
                        vdir, right, up, 
                        vf, rdir,
                        j, num_new_seeds);
                }
            }
            explosion_cooldown--;
        }

        // Debug content
        average_metric /= (num_lines - 1);
        debug[index].index = index;
        // debug[index].maximum_metric = maximum_metric;
        // debug[index].average_metric = average_metric;
        debug[index].maximum_metric = 0.0f;
        debug[index].average_metric = 0.0f;
        debug[index].curand = curand_uniform(&curand_state);

        index += gridDim.x * gridDim.y;
    }
}

bool StreamLineRenderState::trace_streamlines_adaptive(App &app)
{
    // 1. Generate n/2 initial seeds
    int n_initial_seeds = num_seeds / 2;
    if (!generate_seed_points(app, n_initial_seeds))
    {
        std::cerr << "Failed to generate initial adaptive seed points?" << std::endl;
        return false;
    }

    glm::vec3 *seed_points_cuda;
    int *num_seeds_cuda;

    // Allocate maximum amount of seeds
    CHECK_CUDA_ERROR(cudaMalloc(&seed_points_cuda, num_seeds * sizeof(glm::vec3)));
    CHECK_CUDA_ERROR(cudaMemcpy(seed_points_cuda, seed_points.data(), n_initial_seeds * sizeof(glm::vec3), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMalloc(&num_seeds_cuda, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(num_seeds_cuda, &n_initial_seeds, sizeof(int), cudaMemcpyHostToDevice));

    // Allocate debug stuffs
    TraceInfo *debug;
    CHECK_CUDA_ERROR(cudaMalloc(&debug, num_seeds * sizeof(TraceInfo)));
    CHECK_CUDA_ERROR(cudaMemset(debug, 0, num_seeds * sizeof(TraceInfo)));

    float *vbo_data;
    size_t mapped_size;
    CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &streamline_graphics_resource));
    CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer((void **) &vbo_data, &mapped_size, streamline_graphics_resource));
    // Clean the lines
    CHECK_CUDA_ERROR(cudaMemset(vbo_data, 0, mapped_size));

    int trace_start = 0;
    int trace_end = n_initial_seeds - 1;
    int initial_explosion_cooldown = 0;
    
    while (access_data_on_device(num_seeds_cuda, 0) < num_seeds)
    {
        int num_trace = trace_end - trace_start + 1;
        int num_blocks_x = 32;
        int num_blocks_y = (num_trace + (num_blocks_x - 1)) / num_blocks_x;
        // std::cout << "Adaptive trace report: parallel blocks: (" << num_blocks_x << ", " << num_blocks_y << ")" << std::endl;
        std::cout << "Adaptive trace report: " << trace_start << " to " << trace_end << std::endl;

        dim3 num_blocks(num_blocks_x, num_blocks_y, 1);

        trace_and_generate_kernel<<<num_blocks, 1>>>(seed_points_cuda, num_seeds_cuda, num_seeds, num_lines, 
            trace_start, trace_end,
            vbo_data, app.res.vf_tex, simulation_dt, app.ctf_tex_cuda, app.delta_wing_bounding_box,
            use_runge_kutta_4_integrator, seed_point_threshold, adaptive_explosion_radius, num_explosion,
            explosion_cooldown_counter, initial_explosion_cooldown, debug);

        if (initial_explosion_cooldown == 0)
        {
            initial_explosion_cooldown = explosion_cooldown_counter;
        }

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // After tracing, num_seeds_cuda should be changed; now we trace from (trace_start + trace_size) to num_seeds_cuda
        // Stopping strategy:
        // 1. No new seed points are generated;
        // 2. Maximum number of seed points are exceeded.
        trace_start = trace_end + 1;
        trace_end = access_data_on_device(num_seeds_cuda, 0) - 1;

        if (trace_start > trace_end)
        {
            std::cout << "Trace has ended because no new seed points can be generated." << std::endl;
            break;
        }
    }
    if (trace_start <= trace_end)
    {
        // Trace the rest
        trace_end = num_seeds - 1;
        int num_trace = trace_end - trace_start + 1;
        int num_blocks_x = 32;
        int num_blocks_y = (num_trace + (num_blocks_x - 1)) / num_blocks_x;
        std::cout << "Adaptive trace report: " << trace_start << " to " << trace_end << std::endl;
        dim3 num_blocks(num_blocks_x, num_blocks_y, 1);
        constexpr float seed_point_threshold = 10.0f;
        trace_and_generate_kernel<<<num_blocks, 1>>>(seed_points_cuda, num_seeds_cuda, num_seeds, num_lines, 
            trace_start, trace_end,
            vbo_data, app.res.vf_tex, simulation_dt, app.ctf_tex_cuda, app.delta_wing_bounding_box,
            use_runge_kutta_4_integrator, seed_point_threshold, adaptive_explosion_radius, num_explosion,
            explosion_cooldown_counter, explosion_cooldown_counter, debug);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    // Unload seed points back to CPU (to debug)
    // num_seeds_cuda would've been somehow changed
    int final_num_seeds = glm::min(200, access_data_on_device(num_seeds_cuda, 0));
    glm::vec3 *sp_host = new glm::vec3[final_num_seeds];
    CHECK_CUDA_ERROR(cudaMemcpy(sp_host, seed_points_cuda, final_num_seeds * sizeof(glm::vec3), cudaMemcpyDeviceToHost));

    // Push back the rest back to seed_points
    for (int i = n_initial_seeds; i < final_num_seeds; i++)
    {
        seed_points.push_back(sp_host[i]);
        // const glm::vec3 &p = sp_host[i];
        // std::cout << "EXTRA SEED: " << p.x << ", " << p.y << ", " << p.z << std::endl;
    }
    std::cout << "Adaptive trace report: " << (final_num_seeds - n_initial_seeds) << " new seeds." << std::endl;

    TraceInfo *debug_host = new TraceInfo[num_seeds];
    CHECK_CUDA_ERROR(cudaMemcpy(debug_host, debug, num_seeds * sizeof(TraceInfo), cudaMemcpyDeviceToHost));

    // TODO: add/print debug stuffs here...
    // for (int i = 0; i < seed_points.size(); i++)
    // {
    //     std::cout << debug_host[i].average_metric << ", " << debug_host[i].maximum_metric << std::endl;
    // }

    delete[] sp_host;
    delete[] debug_host;
    CHECK_CUDA_ERROR(cudaFree(seed_points_cuda));
    CHECK_CUDA_ERROR(cudaFree(num_seeds_cuda));
    CHECK_CUDA_ERROR(cudaFree(debug));
    CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &streamline_graphics_resource));
    
    return true;
}

__global__ void simplify_streamlines_kernel(int num_streamlines, int num_lines, float *streamline_vbo_data, float distortion_threshold, TraceInfo *debug)
{
    int index = blockIdx.y * gridDim.x + blockIdx.x;
    int streamline_stride = num_lines * 2 * 6;

    while (index < num_streamlines)
    {
        // Memory layout: (POS, COLOR) * 8192
        int start = index * streamline_stride;
        int size = -1;
        float streamline_length = 0.0f;

        // Not every streamline ends at 8192; they must end somewhere in between
        for (int i = 0; i < num_lines; i++)
        {
            float *vertex = &streamline_vbo_data[start + 12 * i];
            if (vertex[0] == 0.0f && vertex[1] == 0.0f && vertex[2] == 0.0f && 
                vertex[6] == 0.0f && vertex[7] == 0.0f && vertex[8] == 0.0f)
            {
                size = i;
                break;
            }
            streamline_length += sqrtf(powf(vertex[6] - vertex[0], 2) + powf(vertex[7] - vertex[1], 2) + powf(vertex[8] - vertex[2], 2));
        }

        if (size == -1)
        {
            size = num_lines;
        }

        // 1. Calculate streamline T(L)
        float *start_vertex = &streamline_vbo_data[start + 12];
        float *end_vertex = &streamline_vbo_data[start + 12 * (size - 1)];
        float streamline_dist = sqrtf(powf(end_vertex[6] - start_vertex[0], 2) + powf(end_vertex[7] - start_vertex[1], 2) + powf(end_vertex[8] - start_vertex[2], 2));
        float global_distortion = streamline_dist == 0.0f ? 0.0f : streamline_length / streamline_dist;

        debug[index].streamline_size = size;
        debug[index].streamline_length = streamline_length;
        debug[index].streamline_dist = streamline_dist;
        debug[index].global_distortion = global_distortion;

        if (size == 0)
        {
            // There is no need to simplify a zero length streamline
            index += gridDim.x * gridDim.y;
            continue;
        }

        // 2. Filter according to T(L) value
        if (global_distortion < distortion_threshold)
        {
            // Zero out the whole thing
            for (int i = 0; i < num_lines; i++)
            {
                float *vertex = &streamline_vbo_data[start + 12 * i];
                memset(vertex, 0, sizeof(float) * 12);
            }
        }

        index += gridDim.x * gridDim.y;
    }
}

bool StreamLineRenderState::simplify_streamlines()
{
    // Simplify streamlines according to, and I quote, "streamline importance".
    int num_streamlines = seed_points.size();
    int num_blocks_x = 32;
    int num_blocks_y = (num_streamlines + (num_blocks_x - 1)) / num_blocks_x;

    float *vbo_data;
    size_t mapped_size;
    CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &streamline_graphics_resource));
    CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer((void **) &vbo_data, &mapped_size, streamline_graphics_resource));

    // Debug info
    TraceInfo *debug;
    CHECK_CUDA_ERROR(cudaMalloc(&debug, num_streamlines * sizeof(TraceInfo)));
    CHECK_CUDA_ERROR(cudaMemset(debug, 0, num_streamlines * sizeof(TraceInfo)));

    dim3 num_blocks(num_blocks_x, num_blocks_y, 1);
    simplify_streamlines_kernel<<<num_blocks, 1>>>(num_streamlines, num_lines, vbo_data, distortion_threshold, debug);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    TraceInfo *debug_host = new TraceInfo[num_streamlines];
    CHECK_CUDA_ERROR(cudaMemcpy(debug_host, debug, num_streamlines * sizeof(TraceInfo), cudaMemcpyDeviceToHost));

    float max_distortion = 0.0f;
    float avg_distortion = 0.0f;
    for (int i = 0; i < num_streamlines; i++)
    {
        // std::cout << i << ": " << debug_host[i].streamline_size << ", " << debug_host[i].streamline_length << ", " << debug_host[i].streamline_dist << std::endl;
        TraceInfo &info = debug_host[i];
        max_distortion = glm::max(max_distortion, info.global_distortion);
        avg_distortion += info.global_distortion;
    }
    avg_distortion /= num_streamlines;
    std::cout << "Max distortion: " << max_distortion << ", average: " << avg_distortion << std::endl;

    delete[] debug_host;
    CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &streamline_graphics_resource));
    CHECK_CUDA_ERROR(cudaFree(debug));
        
    return true;
}

bool StreamLineRenderState::finalize_seed_points(App &app)
{
    if (!seed_points_vao)
    {
        return false;
    }

    glBindBuffer(GL_ARRAY_BUFFER, seed_points_vao->vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(glm::vec3) * seed_points.size(), seed_points.data());
    seed_points_vao->draw_call.size = seed_points.size();

    return true;
}


void StreamLineRenderState::draw_user_controls(App &app)
{
    ImGui::SetNextWindowPos({220.0f, 0.0f}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({app.screen_width - 220.0f, 140}, ImGuiCond_FirstUseEver);
    
    bool should_update = false;

    if (ImGui::Begin("Streamline Controls"))
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

        should_update |= ImGui::Checkbox("Streamline simplification", &do_simplify);
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
        
    }

    ImGui::End();

    if (should_update)
    {
        generate_streamlines(app);
    }
}
