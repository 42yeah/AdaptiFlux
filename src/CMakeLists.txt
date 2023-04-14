set(ADAPTIFLUX_SRC 
    app.cu
    Camera.cpp 
    debug_kernels.cu 
    main.cu 
    Program.cpp 
    VAO.cpp
    Framebuffer.cpp
    renderstates/arrowglyph.cu
    renderstates/lineglyph.cu 
    renderstates/streamline.cu
    renderstates/streamtube.cu)

set_source_files_properties(${ADAPTIFLUX_SRC} PROPERTIES LANGUAGE CUDA)

add_executable(adaptiflux ${ADAPTIFLUX_SRC})
target_include_directories(adaptiflux PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_compile_options(adaptiflux PUBLIC
    -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored)

add_subdirectory(nrrd)
add_subdirectory(shaders)

target_link_libraries(adaptiflux PUBLIC 
    glad 
    imgui
    nrrd
    glm::glm
    glfw
    stb)
