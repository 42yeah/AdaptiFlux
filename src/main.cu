#include <iostream>
#define YYLVV_IMPL
#include "yylvv.cuh"
// #define YYLVV_UI_IMPL
// #include "ui.cuh"
#include "app.cuh"

int main(int argc, char *argv[]) {
    YYLVVRes res;
    if (!initialize_yylvv_contents(argc, argv, res)) {
        std::cerr << "Failed to initialize YYLVV visualizer?" << std::endl;
        return 1;
    }
    App app(res);
    if (!app.valid)
    {
        std::cerr << "Applicationf failed to initialize?" << std::endl;
    }
    else
    {
        app.loop();
    }
    // start_ui(res);
    if (!free_yylvv_resources(res)) {
        std::cerr << "Failed to release YYLVV resources?" << std::endl;
        return 1;
    }
    return 0;
}
