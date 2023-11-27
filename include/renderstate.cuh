#ifndef YYLVV_RENDERSTATE_CUH
#define YYLVV_RENDERSTATE_CUH

#include "yylvv.cuh"
#include <vector>

#define MAX_FRAMERATE_HISTORY 200

class App;

class RenderState
{
public:
    virtual void initialize(App &app) = 0;
    virtual void destroy() = 0;
    virtual void render(App &app) = 0;
    virtual void process_events(App &app) = 0;
    virtual void key_pressed(App &app, int key) = 0;
    virtual void draw_user_controls(App &app) = 0;
};

struct FrameRateInfo
{
    std::vector<float> timestamp;
    std::vector<float> framerate;
    std::vector<float> delta_time;
    float history_xs[MAX_FRAMERATE_HISTORY];
    float history_ys[MAX_FRAMERATE_HISTORY];
    float best_framerate, worst_framerate;
    bool stress_test;
    char stress_test_desc[128];
};

#endif
