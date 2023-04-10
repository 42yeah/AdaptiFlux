//
// Created by Hao Joe on 7/9/2022.
//

#ifndef YYLVV_VECTORFIELD_H
#define YYLVV_VECTORFIELD_H

#include "NRRD.h"
#include <glm/glm.hpp>
#include <iostream>
#include <optional>

struct BBox {
    BBox() = default;
    void enclose(const glm::vec3 &p) {
        min = glm::min(min, p);
        max = glm::max(max, p);
    }

    bool is_inside(const glm::vec3 &p) const {
        return (p.x > min.x && p.y > min.y && p.z > min.z && p.x < max.x && p.y < max.y && p.z < max.z);
    }

    glm::vec3 extend() const {
        return max - min;
    }

    glm::vec3 center() const {
        return (min + max) * 0.5f;
    }

    glm::vec3 min{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
    };
    glm::vec3 max{
        std::numeric_limits<float>::min(),
        std::numeric_limits<float>::min(),
        std::numeric_limits<float>::min()
    };
};

// 3D vector field
class VectorField {
public:
    VectorField(const NRRD &nrrd);
    glm::vec3 at(glm::vec3 pos) const;
    std::optional<glm::vec3> at_discrete(glm::ivec3 indexes) const;

    glm::ivec3 sizes;
    glm::vec3 spacings, spacings_inv;
    BBox bbox;

private:
    std::unique_ptr<glm::vec3[]> vector_field;
    glm::vec3 axis_mins;
};

#endif //YYLVV_VECTORFIELD_H
