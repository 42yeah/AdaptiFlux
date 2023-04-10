//
// Created by Hao Joe on 7/9/2022.
//

#include "VectorField.h"

VectorField::VectorField(const NRRD &nrrd) {
    assert(nrrd.sizes.size() == 4 && nrrd.sizes[0] == 3); // it is user's responsibility to ensure that this is a 3D vector field with 3D directional vectors
    vector_field.reset(new glm::vec3[nrrd.sizes[1] * nrrd.sizes[2] * nrrd.sizes[3]]);
    sizes = {nrrd.sizes[1], nrrd.sizes[2], nrrd.sizes[3]};
    for (int z = 0; z < sizes.z; z++) {
        for (int y = 0; y < sizes.y; y++) {
            for (int x = 0; x < sizes.x; x++) {
                int index = z * sizes.y * sizes.x + y * sizes.x + x;
                vector_field[index] = {
                        nrrd.raw_data[index * 3 + 0],
                        nrrd.raw_data[index * 3 + 1],
                        nrrd.raw_data[index * 3 + 2]
                };
            }
        }
    }
    spacings = {
            nrrd.spacings[0],
            nrrd.spacings[1],
            nrrd.spacings[2]
    };
    axis_mins = {
            nrrd.axis_mins[0],
            nrrd.axis_mins[1],
            nrrd.axis_mins[2]
    };
    spacings_inv = {
            1.0f / spacings.x,
            1.0f / spacings.y,
            1.0f / spacings.z
    }; // for fast division
    // calculate vector field bounding box
    bbox.enclose(axis_mins);
    bbox.enclose(axis_mins + glm::vec3((sizes.x - 1) * spacings.x,
                                       (sizes.y - 1) * spacings.y,
                                       (sizes.z - 1) * spacings.z));
}

glm::vec3 VectorField::at(glm::vec3 pos) const {
    // step 0. check if point is in dead zone
    if (!bbox.is_inside(pos)) {
        return glm::vec3(0.0f);
    }
    // step 1. translate pos back to "canonical space"
    pos -= axis_mins;
    // step 2. transform into discrete space
    pos *= spacings_inv;
    // step 3. perform trilinear interpolation
    glm::ivec3 sa, sb, sc, sd, se, sf, sg, sh;
    sa = glm::ivec3(pos);
    sb = sa + glm::ivec3(1, 0, 0);
    sc = sa + glm::ivec3(1, 0, 1);
    sd = sa + glm::ivec3(0, 0, 1);
    se = sa + glm::ivec3(0, 1, 0);
    sf = se + glm::ivec3(1, 0, 0);
    sg = se + glm::ivec3(1, 0, 1);
    sh = se + glm::ivec3(0, 0, 1);
    glm::vec3 fract = glm::fract(pos);
    glm::vec3 a = *at_discrete(sa), b = *at_discrete(sb), c = *at_discrete(sc),
        d = *at_discrete(sd), e = *at_discrete(se), f = *at_discrete(sf),
        g = *at_discrete(sg), h = *at_discrete(sh);
    glm::vec3 ret_bottom = glm::mix(glm::mix(a, b, fract.x),
                                    glm::mix(d, c, fract.x), fract.y);
    glm::vec3 ret_top = glm::mix(glm::mix(e, f, fract.x),
                                 glm::mix(h, g, fract.x), fract.y);
    return glm::mix(ret_bottom, ret_top, fract.z);
}

std::optional<glm::vec3> VectorField::at_discrete(glm::ivec3 indexes) const {
    if (indexes.x < 0 || indexes.x >= sizes.x || indexes.y < 0 || indexes.y >= sizes.y || indexes.z < 0 || indexes.z >= sizes.z) {
        return std::nullopt;
    }
    return vector_field[indexes.z * sizes.z * sizes.y + indexes.y * sizes.x + indexes.x];
}
