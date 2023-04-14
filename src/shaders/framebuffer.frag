#version 330 core

uniform sampler2D fbo;

in vec2 uv;
out vec4 color;

void main() {
    color = texture(fbo, uv);
}
