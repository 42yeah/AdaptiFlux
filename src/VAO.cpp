//
// Created by admin on 2022/9/15.
//

#include "VAO.h"

VAO::VAO(void *data, int size, GLuint type, const std::vector<VertexAttribPointer> &attribs, GLDrawCall draw_call) :
        draw_call(draw_call) {
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, data, type);
    for (const auto &attrib : attribs) {
        glEnableVertexAttribArray(attrib.position);
        glVertexAttribPointer(attrib.position, attrib.size, attrib.type, attrib.normalize, attrib.stride, attrib.offset);
    }
}

VAO::~VAO() {
    if (vao != GL_NONE) {
        glDeleteVertexArrays(1, &vao);
    }
    if (vbo != GL_NONE) {
        glDeleteBuffers(1, &vbo);
    }
}

std::shared_ptr<VAO> VAO::make_vao(void *data, int size, GLuint type, const std::vector<VertexAttribPointer> &attribs, GLDrawCall draw_call) {
    return std::make_shared<VAO>(data, size, type, attribs, draw_call);
}

void VAO::draw() const {
    glBindVertexArray(vao);
    glDrawArrays(draw_call.type, draw_call.start, draw_call.size);
}
