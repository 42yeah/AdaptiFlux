#version 330 core

in vec3 normal;
in vec3 color;

out vec4 finalColor;

void main() {
    vec3 lightDir = normalize(vec3(1.0, 1.0, 0.0));
    vec3 backLight = normalize(vec3(0.0, 0.0, 1.0));
    float ambient = 0.1;
    float diff = min(1.0, max(0.0, dot(lightDir, normal)));
    float back = min(1.0, max(0.0, dot(backLight, normal)));
    vec3 ambientColor = ambient * color;
    vec3 diffColor = diff * color;
    vec3 backColor = back * 0.5 * color;
    finalColor = vec4(ambientColor + diffColor + backColor, 1.0);
}
