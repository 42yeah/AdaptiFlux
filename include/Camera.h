//
// Created by admin on 2022/9/16.
//

#ifndef YYLVV_CAMERA_H
#define YYLVV_CAMERA_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <optional>

constexpr float pi_over_two = 3.14159265f / 2.0f;
constexpr float infinitesimal = 0.01f;

struct Camera {
    glm::vec3 eye = glm::vec3(0.0f, 0.0f, -1.0f);
    float yaw = 0.0f;
    float sensitivity = 0.005f;

    // half-evaluated components
    float z_near = 0.1f, z_far = 1000.0f;
    float pitch = 0.0f; // pitch & yaw should be looking at (0, 0, 1) when it's neutral
    float speed = 1.0f;

    // evaluated components
    std::optional<glm::dvec2> prev_cursor_pos{};
    glm::vec3 front{0.0f, 0.0f, 1.0f};
    glm::vec3 right{-1.0f, 0.0f, 0.0f}; // left-handed coordinate
    glm::vec3 center{0.0f, 0.0f, 0.0f};
    glm::mat4 view{1.0f}, perspective{1.0f};

    void update_components(int screen_width, int screen_height);
};


#endif //YYLVV_CAMERA_H
