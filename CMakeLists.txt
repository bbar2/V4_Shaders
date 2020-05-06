cmake_minimum_required(VERSION 3.15)

# To use template project:
#  Copy template project folder
#  Change VULKAN_PROJECT_NAME
set(VULKAN_PROJECT_NAME VulkanShaders)
set(VULKAN_SDK_PATH /home/barry/Vulkan/1.2.131.2/x86_64)
set(CMAKE_CXX_STANDARD 17)

project(${VULKAN_PROJECT_NAME})

# CMake function for simple compile of shader.xxx to xxx.spv
function(compile_shader SHADER)
    add_custom_command( TARGET ${VULKAN_PROJECT_NAME} PRE_BUILD
            COMMAND ${VULKAN_SDK_PATH}/bin/glslc shader.${SHADER} -o ${SHADER}.spv
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/shaders
            COMMENT "Compiling shader.${SHADER} to ${SHADER}.spv")
endfunction()

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
compile_shader("vert")
compile_shader("frag")

# Look at some built-in vars
message(CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR})
message(CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR})
message(CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME})
message(CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE})

