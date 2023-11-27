#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 aColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 perspective;

out vec3 pos;
out vec3 normal;
out vec3 color;

void main() {
    vec4 modelPos = model * vec4(aPos, 1.0);
    gl_Position = perspective * view * modelPos;
    pos = vec3(modelPos);
    normal = vec3(model * vec4(aNormal, 0.0));
    color = vec3(aColor);
}
