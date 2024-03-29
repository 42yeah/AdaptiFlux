#ifndef APP_CUH
#define APP_CUH

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "yylvv.cuh"
#include "VAO.h"
#include "Program.h"
#include "Camera.h"
#include <VectorField.h> // for bounding box
#include <vector>
#include "renderstate.cuh"
#include "Framebuffer.h"

class RenderState;

class App
{
public:
    App(YYLVVRes &res);
    ~App();

    App(const App &) = delete;
    App(App &&) = delete;

    void loop();

    bool valid;

private:
    bool init();

    static void key_callback_glfw(GLFWwindow *window, int key, int scancode, int action, int mods);
    static void cursor_pos_callback_glfw(GLFWwindow *window, double xpos, double ypos);
    static void window_size_callback_glfw(GLFWwindow *window, int width, int height);
    void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods);
    void cursor_pos_callback(GLFWwindow *window, double xpos, double ypos);
    void window_size_callback(GLFWwindow *window, int width, int height);

    void align_camera();
    void handle_continuous_key_events();

    void draw_delta_wing() const;

    void switch_state(std::shared_ptr<RenderState> new_state);

    // User controls
    void draw_user_controls();
    void framerate_layer();
    void set_user_interface_mode(bool new_ui_mode);

    // Favourite camera pose
    void favorite_camera_pose() const;
    void restore_camera_pose();
    
    //
    // Take a screenshot.
    // Screenshots will be automatically saved to figs/ folder (which will be created, if it doesn't exist)
    // With the following name: <vectorfield_name>_<resolution>_<time>.jpg
    //
    bool screenshot() const;

    // Debug sample vector field
    void debug_vf() const;

public:
    YYLVVRes &res;
    GLFWwindow *window;

    // UI resources
    std::shared_ptr<VAO> bounding_box_vao;
    std::shared_ptr<Program> bounding_box_program;
    Camera camera;

    std::shared_ptr<RenderState> render_state; // current RenderState - supports line glyphs, etc.
    int screen_width, screen_height;
    double last_instant; // time-related variable
    float delta_time; // elapsed time from last frame

    // delta wing related stuffs
    BBox delta_wing_bounding_box;
    std::shared_ptr<VAO> delta_wing_vao;
    std::shared_ptr<Program> delta_wing_program;

    // shadow mapping related
    std::shared_ptr<VAO> shadow_floor;
    std::shared_ptr<Program> shadow_floor_program;

    // color transfer function texture
    cudaArray_t ctf_data_cuda;
    cudaTextureObject_t ctf_tex_cuda;

    // global framebuffer (for sceenshot & hires rendering)
    std::unique_ptr<Framebuffer> framebuffer;
    std::shared_ptr<VAO> rect_vao;
    std::shared_ptr<Program> framebuffer_render_program;
    bool custom_resolution;
    glm::ivec2 custom_resolution_size;

    // UI controls
    bool user_interface_mode;
    int visualization_mode;
    bool should_draw_bounding_box;
    bool should_draw_delta_wing;
    bool should_draw_shadow;

    // Frame rate history, and other UI information
    float elapsed;
    FrameRateInfo framerate_history;
    float framerate_sum;
};

#endif // APP_CUH
