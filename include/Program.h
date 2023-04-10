//
// Created by Hao Joe on 8/9/2022.
//

#ifndef YYLVV_PROGRAM_H
#define YYLVV_PROGRAM_H

#include <map>
#include <iostream>
#include <glad/glad.h>

class Program;
using ProgramPtr = std::shared_ptr<Program>;

class Program {
public:
    Program();
    ~Program();
    Program(const std::string &vertex_path, const std::string &fragment_path);
    bool link(const std::string &vertex_path, const std::string &fragment_path);
    GLuint operator[](const std::string &uniform_name) const;
    GLuint at(const std::string &uniform_name) const;
    void use() const;
    static ProgramPtr make_program(const std::string &vertex_path, const std::string &fragment_path);

    bool valid; // it is user's responsibility to check the validity of program

private:
    mutable std::map<std::string, GLuint> locations;
    GLuint program;
};


#endif //YYLVV_PROGRAM_H
