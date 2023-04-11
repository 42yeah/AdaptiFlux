#ifndef YYLVV_STREAMTUBE_CUH
#define YYLVV_STREAMTUBE_CUH

#include "renderstate.cuh"
#include "streamline.cuh"
#include "VectorField.h"
#include "VAO.h"
#include "Program.h"
#include <vector>

// StreamTubeRenderState basically bases itself on streamlines,
// and just magic tubes out of thin air.
class StreamTubeRenderState : public StreamLineRenderState 
{
public:
    StreamTubeRenderState();
    ~StreamTubeRenderState();

    virtual void initialize(App &app) override;
    virtual void destroy() override;
    virtual void render(App &app) override;
    virtual void process_events(App &app) override;
    virtual void key_pressed(App &app, int key) override;
    virtual void draw_user_controls(App &app) override;

private:
    bool allocate_graphics_resources();
    bool generate_streamtubes();
    bool export_streamtube_vbo_as_obj(const std::string &path);

    // Some test code
    bool generate_bare_bones_streamline();
    bool test_streamtube_generation();

    std::shared_ptr<VAO> streamtube_vao;
    std::shared_ptr<Program> streamtube_program;
    cudaGraphicsResource *streamtube_graphics_resource;

    float streamtube_radius;
    bool shadow_mapping;
};

#endif
