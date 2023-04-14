#ifndef YYLVV_FRAMEBUFFER_H
#define YYLVV_FRAMEBUFFER_H

#include <glad/glad.h>
#include <iostream>
#include "Program.h"
#include "VAO.h"

class Framebuffer
{
public:
    Framebuffer(int width, int height);
    ~Framebuffer();

    bool screenshot(const std::string &path) const;

    void use();
    void done();
    void resize(int new_width, int new_height);

    // 
    // Draw on a <VAO> using <Shader>. 
    // This simply calls vao->draw() with texture_gl bound as GL_TEXTURE0.
    // Ideally, VAO should be a full-screen rect, while shader should just be a simple texture sample call.
    //
    void draw(std::shared_ptr<VAO> vao, std::shared_ptr<Program> program, 
        const std::string &uniform_name, int texture_id = 0);

    GLuint get_texture() const;

    bool render_test_buffer();

private:
    GLuint framebuffer_gl;
    GLuint texture_gl;
    GLuint rbo_gl;
    int width, height;

    // Previous viewport configurations
    bool in_use;
    int last_viewport_conf[4];
};

#endif 
