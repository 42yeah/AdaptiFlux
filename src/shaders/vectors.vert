#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 perspective;

out vec3 color;

void main() {
    gl_Position = perspective * view * model * vec4(aPos, 1.0);
    color = vec3(aColor);
}
