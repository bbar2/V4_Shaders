cmake_minimum_required(VERSION 3.15)

# To use template project:
#  Copy template project folder
#  Change VULKAN_PROJECT_NAME
set(VULKAN_PROJECT_NAME VulkanShaders)

project(${VULKAN_PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)

# enable pkg-config actions via CMake GLFW3_STATIC_LIBRARIES flag.
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW3 REQUIRED glfw3)

set(VULKAN_SDK_PATH /home/barry/Vulkan/1.2.131.2/x86_64)

include_directories(${VULKAN_SDK_PATH}/include)
link_directories(${VULKAN_SDK_PATH}/lib)

# pkg-config included -L/usr/local/lib, which is not in GLFW3_STATIC_LIBRARIES
link_directories(/usr/local/lib)
link_libraries(${GLFW3_STATIC_LIBRARIES})
message(GLFW3_STATIC_LIBRARIES: ${GLFW3_STATIC_LIBRARIES})

link_libraries(vulkan)

# compile shaders
#execute_process(COMMAND shaders/compile.sh WORKING_DIRECTORY shaders)
#add_custom_target(vert.spv ALL DEPENDS shaders/shader.vert)
#add_custom_command(
#        OUTPUT shaders/vert.spv
#        COMMAND ${VULKAN_SDK_PATH}/bin/glslc shader.vert -o vert.spv
#        MAIN_DEPENDENCY shader.vert
#        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/shaders
#        COMMENT this is a test
#)

add_executable(${VULKAN_PROJECT_NAME} main.cpp)

# Look at some built-in vars
message(CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR})
message(CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR})
message(CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME})
message(CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE})

