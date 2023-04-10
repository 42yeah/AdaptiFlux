//
// Created by admin on 2022/9/15.
//

#ifndef YYLVV_VAO_H
#define YYLVV_VAO_H

#include <glad/glad.h>
#include <iostream>
#include <vector>

struct VertexAttribPointer {
    GLuint position;
    GLuint size;
    GLuint type;
    GLuint normalize;
    GLuint stride;
    void *offset;

    VertexAttribPointer(GLuint pos, GLuint sz, GLuint t, GLuint nor, GLuint stri, void *off) :
            position(pos),
            size(sz),
            type(t),
            normalize(nor),
            stride(stri),
            offset(off) {
    }
};

struct GLDrawCall {
    GLuint type;
    GLuint start;
    GLuint size;

    GLDrawCall(GLuint t, GLuint st, GLuint sz) :
            type(t),
            start(st),
            size(sz) {
    }
};

// managed OpenGL VAO
struct VAO {
    static std::shared_ptr<VAO> make_vao(void *data, int size, GLuint type, const std::vector<VertexAttribPointer> &attribs, GLDrawCall draw_call);
    VAO(void *data, int size, GLuint type, const std::vector<VertexAttribPointer> &attribs, GLDrawCall draw_call);
    VAO(const VAO &) = delete;
    VAO(VAO &&) = delete;
    ~VAO();
    void draw() const;

    GLuint vao, vbo;
    GLDrawCall draw_call;
};

#endif //YYLVV_VAO_H
