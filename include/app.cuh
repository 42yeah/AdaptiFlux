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
#include "renderstate.cuh"

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
    void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods);
    void cursor_pos_callback(GLFWwindow *window, double xpos, double ypos);

    void align_camera();
    void handle_continuous_key_events();

    void draw_delta_wing() const;

    void switch_state(std::shared_ptr<RenderState> new_state);

    // User controls
    void draw_user_controls();
    void set_user_interface_mode(bool new_ui_mode);

    // Favourite camera pose
    void favorite_camera_pose() const;
    void restore_camera_pose();

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

    // UI controls
    bool user_interface_mode;
    int visualization_mode;
    bool should_draw_bounding_box;
    bool should_draw_delta_wing;
    bool should_draw_shadow;
};

#endif // APP_CUH
