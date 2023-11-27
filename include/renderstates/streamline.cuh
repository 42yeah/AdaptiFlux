#ifndef YYLVV_STREAMLINE_CUH
#define YYLVV_STREAMLINE_CUH

#include "renderstate.cuh"
#include "VectorField.h"
#include "VAO.h"
#include "Program.h"
#include <vector>

class StreamLineRenderState : public RenderState 
{
public:
    StreamLineRenderState();
    ~StreamLineRenderState();

    virtual void initialize(App &app) override;
    virtual void destroy() override;
    virtual void render(App &app) override;
    virtual void process_events(App &app) override;
    virtual void key_pressed(App &app, int key) override;
    virtual void draw_user_controls(App &app) override;

protected:
    bool allocate_graphics_resources();
    bool generate_streamlines(App &app);
    bool generate_seed_points(App &app, int num_seeds);
    bool generate_seed_points_delta_wing(const BBox &bbox, int num_seeds);
    bool generate_seed_points_line(glm::vec3 a, glm::vec3 b, int num_seeds);
    bool generate_seed_points_rect(glm::vec3 a, glm::vec3 b, int num_seeds);
    bool generate_seed_points_spherical(glm::vec3 c, float r, int num_seeds);
    bool trace_streamlines(App &app);
    bool finalize_seed_points(App &app);

    bool trace_streamlines_adaptive(App &app);
    bool simplify_streamlines();

    // seed point generation configurations
    int num_seeds, num_lines;
    float simulation_dt;
    std::vector<glm::vec3> seed_points;

    // streamline variables
    std::shared_ptr<VAO> streamline_vao;
    std::shared_ptr<Program> streamline_program;
    cudaGraphicsResource *streamline_graphics_resource;
    bool use_runge_kutta_4_integrator;
    float seeding_plane_x;
    bool adaptive_mode;

    // adaptive mode streamline variables
    float adaptive_explosion_radius;
    int num_explosion;
    int explosion_cooldown_counter;
    float seed_point_threshold;

    // streamline simplification
    bool do_simplify;
    float distortion_threshold;

    // true == use delta wing recommended strat
    int seed_points_strategy;
    glm::vec3 seed_begin;
    glm::vec3 seed_end;
    float seed_radius;

    // seed point rendering
    bool render_seed_points;
    float point_size;
    std::shared_ptr<VAO> seed_points_vao;
    std::shared_ptr<Program> seed_points_program;
};

#endif
