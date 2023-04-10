//
// Created by admin on 2022/9/16.
//

#include "Camera.h"

void Camera::update_components(int screen_width, int screen_height) {
    // pitch should be clamped at
    pitch = glm::clamp(pitch, -pi_over_two + infinitesimal, pi_over_two - infinitesimal);
    front = glm::vec3(cosf(pitch) * sinf(yaw), sinf(pitch), cosf(pitch) * cosf(yaw));
    center = eye + front;
    right = glm::cross(front, glm::vec3(0.0f, 1.0f, 0.0f));
    view = glm::lookAt(eye, center, glm::vec3(0.0f, 1.0f, 0.0f));
    perspective = glm::perspective(glm::radians(60.0f), (float) screen_width / screen_height, z_near, z_far);
}
