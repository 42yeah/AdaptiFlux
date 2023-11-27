#version 330 core

in vec3 pos;
in vec3 normal;
in vec3 color;

uniform vec3 camera;
uniform vec3 lightDir;

out vec4 finalColor;

void main() {
    float ambient = 0.2;
    float diffuse = max(dot(normal, lightDir), 0.0);

    vec3 reflected = reflect(-lightDir, normal);
    vec3 camDir = normalize(camera - pos);
    float specular = pow(max(dot(reflected, camDir), 0.0), 32.0) * 0.5;

    float bottomLight = max(dot(normal, vec3(0.0, -1.0, 0.0)), 0.0) * 0.2;
    float frontLight = -max(dot(normal, camDir), 0.0) * 0.1;
    float fringe = pow(1.0 - max(dot(normal, camDir), 0.0), 32.0) * 0.5;
    float backLight = max(dot(normal, vec3(0.0, 0.0, -1.0)), 0.0) * 0.1;
    float domeLight = max(dot(normal, vec3(0.0, 0.0, -1.0)), 0.0) * 0.2;

    vec3 lightColor = min(1.0, max(ambient + diffuse + bottomLight + frontLight + fringe + backLight + domeLight, 0.0)) * vec3(0.92, 0.96, 0.99) * color;
    vec3 specularColor = specular * vec3(1.0, 1.0, 1.0) * color;

    vec3 combined = lightColor + specularColor;
    // HDR 
    // combined = combined / (combined + vec3(1.0));

    // gamma
    combined = pow(combined.rgb, vec3(1.0 / 2.2));
    
    finalColor = vec4(combined, 1.0);
}
