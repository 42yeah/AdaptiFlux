#ifndef YYLVV_ARROWGLYPH_CUH
#define YYLVV_ARROWGLYPH_CUH

#include "renderstate.cuh"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "VAO.h"
#include "Program.h"

class ArrowGlyphRenderState : public RenderState {
public:
    ArrowGlyphRenderState();
    virtual void initialize(App &app) override;
    virtual void destroy() override;
    virtual void render(App &app) override;
    virtual void process_events(App &app) override;
    virtual void key_pressed(App &app, int key) override;
    virtual void draw_user_controls(App &app) override;

private:
    void initialize_arrow_glyph_resources();
    bool generate_arrow_glyphs(App &app, float z);
    void clamp_visualizing_z(App &app);

    // Vector field arrow glyph configurations
    int num_arrows_x, num_arrows_y;
    std::shared_ptr<VAO> arrow_glyph_vao;
    std::shared_ptr<Program> arrow_glyph_program;
    cudaGraphicsResource *arrow_glyph_graphics_resource;
    float vector_length;
    float current_visualizing_z;
    bool visualize_xy;
};

#endif
