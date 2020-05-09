cmake_minimum_required(VERSION 3.15)
message(CMAKE_VERSION: ${CMAKE_VERSION})

# To use template project:
#  Copy template project folder
#  Change VULKAN_PROJECT_NAME
set(VULKAN_PROJECT_NAME VulkanShaders)
set(VULKAN_SDK_PATH /home/barry/Vulkan/1.2.131.2/x86_64)
set(CMAKE_CXX_STANDARD 17)

project(${VULKAN_PROJECT_NAME})

# enable pkg-config actions via CMake GLFW3_STATIC_LIBRARIES flag.
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW3 REQUIRED glfw3)

include_directories(${VULKAN_SDK_PATH}/include)
link_directories(${VULKAN_SDK_PATH}/lib)

# pkg-config included -L/usr/local/lib, which is not in GLFW3_STATIC_LIBRARIES
link_directories(/usr/local/lib)
link_libraries(${GLFW3_STATIC_LIBRARIES})
message(GLFW3_STATIC_LIBRARIES: ${GLFW3_STATIC_LIBRARIES})

link_libraries(vulkan)

add_executable(${VULKAN_PROJECT_NAME} main.cpp)

# this seems to do it all, but seems overly complex.
# -- add_custom_command: OUTPUT variety to create xxx.spv file dependent on the shader.xxx files
# -- add_custom_target: new target dependent on xxx.spv files
# -- add_dependencies: to associate my_shaders target with main ${VULCAN_PROJECT_NAME} target
function(compile_shader NAME)
    # create a variable of all compiled shader file names, for add_custom_target
    set(SPV_FILES ${SPV_FILES} ${CMAKE_SOURCE_DIR}/shaders/${NAME}.spv PARENT_SCOPE)

    # Build a command to compile each xxx.spv from shader.xxx source
    add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/shaders/${NAME}.spv
            COMMAND ${VULKAN_SDK_PATH}/bin/glslc shader.${NAME} -o ${NAME}.spv
            MAIN_DEPENDENCY shaders/shader.${NAME}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/shaders
            COMMENT "Compiling shader.${NAME} to ${NAME}.spv")
endfunction()

# call these before add_custom_target, to build the SPV_FILES variable
compile_shader(vert)
compile_shader(frag)
add_custom_target(my_shaders DEPENDS ${SPV_FILES})
add_dependencies(${VULKAN_PROJECT_NAME} my_shaders)

# Look at some built-in vars
message(CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR})
message(CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR})
message(CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME})
message(CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE})
message(SHADER_SOURCE_FILES: ${SHADER_SOURCE_FILES})
