//
// Created by Hao Joe on 8/9/2022.
//

#include "Program.h"
#include <fstream>
#include <sstream>
#include <optional>

std::optional<GLuint> compile(GLuint type, const std::string &path) {
    std::ifstream reader(path);
    if (!reader.good()) {
        return std::nullopt;
    }
    GLuint shader = glCreateShader(type);
    std::stringstream ss;
    ss << reader.rdbuf();
    std::string str = ss.str();
    const char *raw = str.c_str();
    glShaderSource(shader, 1, &raw, nullptr);
    glCompileShader(shader);
    GLint state = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &state);
    if (state == GL_FALSE) {
        char log[512] = {0};
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::cerr << "Failed to compile " << path << "?: " << log << std::endl;
        glDeleteShader(shader);
        return std::nullopt;
    }
    return shader;
}

std::optional<GLuint> link(GLuint vertex_shader, GLuint fragment_shader) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);
    GLint state = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &state);
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    if (state == GL_FALSE) {
        char log[512] = {0};
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        std::cerr << "Failed to link program?: " << log << std::endl;
        glDeleteProgram(program);
        return std::nullopt;
    }
    return program;
}

Program::Program() : valid(false), program(GL_NONE) {

}

Program::~Program() {
    if (valid) {
        glDeleteProgram(program);
    }
}

Program::Program(const std::string &vertex_path, const std::string &fragment_path) : valid(false), program(GL_NONE) {
    link(vertex_path, fragment_path);
}

bool Program::link(const std::string &vertex_path, const std::string &fragment_path) {
    if (valid) {
        glDeleteProgram(program);
    }
    valid = false;
    auto vertex_shader = compile(GL_VERTEX_SHADER, vertex_path);
    if (!vertex_shader) { return false; }
    auto fragment_shader = compile(GL_FRAGMENT_SHADER, fragment_path);
    if (!fragment_shader) { return false; }
    auto program_opt = ::link(*vertex_shader, *fragment_shader);
    if (!program_opt) {
        return false;
    }
    program = *program_opt;
    valid = true;
    return true;
}

GLuint Program::operator[](const std::string &uniform_name) const {
    if (!valid) {
        return GL_NONE;
    }
    auto pos = locations.find(uniform_name);
    if (pos == locations.end()) {
        GLint loc = glGetUniformLocation(program, uniform_name.c_str());
        if (loc < 0) {
            std::cerr << "Cannot find uniform name: " << uniform_name << "?" << std::endl;
        }
        locations[uniform_name] = loc;
        return (GLuint) loc;
    }
    return pos->second;
}

void Program::use() const {
    if (!valid) {
        std::cerr << "Warning: using invalid program?" << std::endl;
        return;
    }
    glUseProgram(program);
}

ProgramPtr Program::make_program(const std::string &vertex_path, const std::string &fragment_path) {
    std::shared_ptr<Program> program = std::make_shared<Program>();
    if (!program->link(vertex_path, fragment_path)) {
        return nullptr;
    }
    return program;
}

GLuint Program::at(const std::string &uniform_name) const {
    return this->operator[](uniform_name);
}
