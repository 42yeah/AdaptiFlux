#ifndef YYLVV_RENDERSTATE_CUH
#define YYLVV_RENDERSTATE_CUH

#include "yylvv.cuh"

class App;

class RenderState {
public:
    virtual void initialize(App &app) = 0;
    virtual void destroy() = 0;
    virtual void render(App &app) = 0;
    virtual void process_events(App &app) = 0;
    virtual void key_pressed(App &app, int key) = 0;
    virtual void draw_user_controls(App &app) = 0;
};

#endif
