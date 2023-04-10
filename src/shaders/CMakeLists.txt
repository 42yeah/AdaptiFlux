set(SHADERS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(DEST_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(
    shaders ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${SHADERS_SOURCE_DIR} ${DEST_DIR}
    COMMENT "Deploying the shaders folder."
)
