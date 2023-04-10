#ifndef PLAINTEXT_H
#define PLAINTEXT_H

#include <iostream>
#include <vector>
#include <glm/glm.hpp>

class PlainText
{
public:
    PlainText();
    ~PlainText() = default;

    bool load_from_file(const std::string &path);
    int raw_data_size() const;

    glm::ivec3 sizes;
    std::unique_ptr<float[]> raw_data;
};


#endif // PLAINTEXT_H
