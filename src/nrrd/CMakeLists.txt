add_library(nrrd NRRD.cpp NRRD.h VectorField.cpp VectorField.h PlainText.cpp)
target_link_libraries(nrrd PUBLIC
        glm::glm)
target_include_directories(nrrd PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR})
