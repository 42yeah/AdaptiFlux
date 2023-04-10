//
// Created by Hao Joe on 7/9/2022.
//

#ifndef YYLVV_NRRD_H
#define YYLVV_NRRD_H

#include <iostream>
#include <vector>

class VectorField;

// custom NRRD loader with NO big endian support
class NRRD {
public:
    NRRD();
    ~NRRD() = default;
    bool load_from_file(const std::string &path);
    int raw_data_size() const;

    std::vector<int> sizes;

    std::unique_ptr<float[]> raw_data;
    std::unique_ptr<float[]> spacings, axis_mins;

private:
    bool parse_line(const std::string &line);

    friend class VectorField;
};


#endif //YYLVV_NRRD_H
