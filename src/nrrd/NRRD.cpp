//
// Created by Hao Joe on 7/9/2022.
//

#include "NRRD.h"
#include <fstream>
#include <iostream>
#include <string>
#include <functional>

NRRD::NRRD() : raw_data(nullptr) {
}

void split_string(const std::string &str, std::function<void(int idx, const std::string &res)> func) {
    int idx = 0;
    std::string value = str;
    while (true) {
        auto space_pos = value.find(' '); // find next whitespace
        std::string tok;
        if (space_pos == std::string::npos) {
            tok = value;
        } else {
            tok = value.substr(0, space_pos);
        }
        func(idx, tok);
        value = value.substr(space_pos + 1);
        idx++;
        if (space_pos == std::string::npos) {
            return;
        }
    }
}

bool NRRD::load_from_file(const std::string &path) {
    std::ifstream reader(path, std::ios::binary);
    if (!reader.good()) {
        return false;
    }
    char header[4];
    reader.read(header, sizeof(header));
    if (std::strncmp(header, "NRRD", 4) != 0) {
        // invalid file
        return false;
    }
    std::string line;
    std::getline(reader, line); // skip the first line
    while (true) {
        std::getline(reader, line);
        if (line.empty()) {
            break;
        }
        if (!parse_line(line)) {
            return false;
        }
    }
    if (sizes.size() == 0) {
        return false;
    }
    int rd_size = raw_data_size();
    for (int i = 0; i < rd_size; i++) {
        reader.read((char *) &raw_data[i], sizeof(float));
        if (reader.eof()) {
            // premature EOF
            return false;
        }
    }
    return true;
}

bool NRRD::parse_line(const std::string &line) {
    auto pos = line.find(':');
    if (pos == std::string::npos) {
        return false; // immature end to NRRD file parsing
    }
    auto key = line.substr(0, pos);
    auto value = line.substr(pos + 1);
    // trim value
    auto nonspace = value.find_first_not_of(' ');
    if (nonspace == std::string::npos) {
        return false;
    }
    value = value.substr(nonspace);
    // fail the parsing if type is not float
    if (key == "type") {
        if (value != "float") {
            return false;
        }
    } else if (key == "dimension") {
        int dimension = std::stoi(value);
        if (dimension == 1) {
            return false;
        }
        sizes.resize(dimension);
        spacings.reset(new float[dimension - 1]);
        axis_mins.reset(new float[dimension - 1]);
    } else if (key == "sizes") {
        int raw_data_size = 1;
        for (int i = 0; i < sizes.size(); i++) {
            if (value.empty()) {
                return false;
            }
            auto space_pos = value.find(' '); // find next whitespace
            std::string tok;
            if (space_pos == std::string::npos) {
                tok = value;
            } else {
                tok = value.substr(0, space_pos);
            }
            sizes[i] = std::stoi(tok);
            value = value.substr(space_pos + 1);
            raw_data_size *= sizes[i];
        }
        raw_data.reset(new float[raw_data_size]);
    } else if (key == "spacings") {
        split_string(value, [&](int i, const auto &s) {
            if (i == 0) {
                return;
            }
            spacings[i - 1] = std::stof(s);
        });
    } else if (key == "axis mins") {
        split_string(value, [&](int i, const auto &s) {
            if (i == 0) {
                return;
            }
            axis_mins[i - 1] = std::stof(s);
        });
    } else if (key == "labels") {
        // ignore
    } else if (key == "endian") {
        if (value != "little") {
            return false;
        }
    } else if (key == "encoding") {
        if (value != "raw") {
            return false;
        }
    }
    return true;
}

int NRRD::raw_data_size() const {
    int ret = 1;
    for (auto s : sizes) {
        ret *= s;
    }
    return ret;
}
