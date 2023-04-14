#include "Framebuffer.h"
#include <cassert>
#include <stb_image_write.h>
#include "utils.cuh"
#include "VAO.h"
#include "Program.h"

Framebuffer::Framebuffer(int width, int height) : width(width), height(height), in_use(false)
{
    // Generate texture
    glGenTextures(1, &texture_gl);
    glBindTexture(GL_TEXTURE_2D, texture_gl);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

    // Generate RBO
    glGenRenderbuffers(1, &rbo_gl);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo_gl);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);

    glGenFramebuffers(1, &framebuffer_gl);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_gl);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_gl, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo_gl);

    glBindFramebuffer(GL_FRAMEBUFFER, GL_NONE);
    CHECK_OPENGL_ERRORS();
}

Framebuffer::~Framebuffer()
{
    glDeleteFramebuffers(1, &framebuffer_gl);
    glDeleteTextures(1, &texture_gl);
    glDeleteRenderbuffers(1, &rbo_gl);

    CHECK_OPENGL_ERRORS();
}

bool Framebuffer::screenshot(const std::string &path) const
{
    std::unique_ptr<unsigned int[]> data = std::make_unique<unsigned int[]>(width * height * 3 * sizeof(unsigned int));
    glBindTexture(GL_TEXTURE_2D, texture_gl);

    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, data.get());

    stbi_flip_vertically_on_write(true);
    int ret = stbi_write_jpg(path.c_str(), width, height, 3, data.get(), 100);
    if (ret == 0)
    {
        return false;
    }

    CHECK_OPENGL_ERRORS();
    return true;
}

void Framebuffer::use()
{
    in_use = true;
    glGetIntegerv(GL_VIEWPORT, last_viewport_conf);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_gl);
    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    CHECK_OPENGL_ERRORS();
}

void Framebuffer::done()
{
    in_use = false;
    glBindFramebuffer(GL_FRAMEBUFFER, GL_NONE);
    glViewport(last_viewport_conf[0], last_viewport_conf[1], last_viewport_conf[2], last_viewport_conf[3]);
    CHECK_OPENGL_ERRORS();
}

void Framebuffer::resize(int new_width, int new_height)
{
    width = new_width;
    height = new_height;

    glBindTexture(GL_TEXTURE_2D, texture_gl);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

    glBindRenderbuffer(GL_RENDERBUFFER, rbo_gl);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);

    if (in_use)
    {
        glViewport(0, 0, width, height);
    }

    CHECK_OPENGL_ERRORS();
}

GLuint Framebuffer::get_texture() const
{
    return texture_gl;
}

bool Framebuffer::render_test_buffer()
{
    float triangle_data[] = 
    {
        0.0f, 0.0f, 0.0f,
        0.5f, 0.0f, 0.0f,
        0.0f, 0.5f, 0.0f
    };

    std::shared_ptr<VAO> triangle = std::make_shared<VAO>(triangle_data, sizeof(triangle_data), GL_STATIC_DRAW, 
        std::vector<VertexAttribPointer>(
            {
                VertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, nullptr)
            }
        ),
        GLDrawCall(GL_TRIANGLES, 0, 3));

    std::shared_ptr<Program> program = Program::make_program("shaders/simple.vert", "shaders/simple.frag");
    if (!program || !program->valid)
    {
        std::cerr << "Invalid simple shaders?" << std::endl;
        return false;
    }

    use();
    glClearColor(1.0f, 0.5f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    program->use();
    triangle->draw();

    if (!screenshot("target.jpg"))
    {
        std::cerr << "Cannot screenshot to target.jpg." << std::endl;
        return false;
    }

    resize(1920, 1080);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    program->use();
    triangle->draw();
    if (!screenshot("target_hd.jpg"))
    {
        std::cerr << "Cannot screenshot to target_hd.jpg." << std::endl;
        return false;
    }

    CHECK_OPENGL_ERRORS();
    done();

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    return true;
}

void Framebuffer::draw(std::shared_ptr<VAO> vao, std::shared_ptr<Program> program, const std::string &uniform_name, int texture_id)
{
    program->use();
    glBindTexture(GL_TEXTURE_2D, texture_gl);
    glActiveTexture(GL_TEXTURE0 + texture_id);
    glUniform1i(program->at(uniform_name), texture_id);
    vao->draw();

    CHECK_OPENGL_ERRORS();
}

glm::ivec2 Framebuffer::get_size() const
{
    return { width, height };
}
