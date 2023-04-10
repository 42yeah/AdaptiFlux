#include "PlainText.h"
#include <fstream>


PlainText::PlainText() : sizes(0, 0, 0), raw_data(nullptr)
{

}

bool PlainText::load_from_file(const std::string &path)
{
    std::ifstream reader(path);

    if (!reader.good())
    {
        std::cerr << "Failed to parse plain text vector field from path: " << path << "?" << std::endl;
        return false;
    }

    reader >> sizes.x >> sizes.y >> sizes.z;
    int expected_size = sizes.x * sizes.y * sizes.z * 3;
    int real_size = 0;
    std::cout << "Allocating expected size of " << expected_size << " floats" << std::endl;

    raw_data.reset(new float[expected_size]);

    for (int i = 0; i < expected_size; i++)
    {
        if (!(reader >> raw_data[i]))
        {
            break;
        }
        real_size++;
    }
    std::cout << "Plain text loading done. Size: " << real_size << "/" << expected_size << std::endl;

    int to_go = 0;
    float trash;
    while (reader >> trash)
    {
        to_go++;
    }
    if (to_go != 0)
    {
        std::cerr << "Warning: " << to_go << " floats unread?" << std::endl;
    }

    return real_size == expected_size;
}
