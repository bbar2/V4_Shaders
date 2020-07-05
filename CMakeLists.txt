cmake_minimum_required(VERSION 3.15)
message(CMAKE_VERSION: ${CMAKE_VERSION})

# To use template project:
#  Copy template project folder
#  Change VULKAN_PROJECT_NAME
set(VULKAN_PROJECT_NAME VulkanShaders)
project(${VULKAN_PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
message(CMAKE_C_COMPILER: ${CMAKE_C_COMPILER})

# Set all paths based on current operating system
if (UNIX)
    # Vulkan SDK on Linux
    set(VULKAN_SDK_PATH /home/barry/Vulkan/1.2.131.2/x86_64)
    include_directories(${VULKAN_SDK_PATH}/include)
    link_directories(${VULKAN_SDK_PATH}/lib)
    link_libraries(vulkan)

    # enable pkg-config actions via CMake GLFW3_STATIC_LIBRARIES flag.
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GLFW3 REQUIRED glfw3)
    message(GLFW3_STATIC_LIBRARIES: ${GLFW3_STATIC_LIBRARIES})

    # pkg-config included -L/usr/local/lib, which is not in GLFW3_STATIC_LIBRARIES
    link_directories(/usr/local/lib)
    link_libraries(${GLFW3_STATIC_LIBRARIES})

else()
    # Vulkan SDK on Windows
    set(VULKAN_SDK_PATH c:/VulkanSDK/1.2.135.0)
    include_directories(${VULKAN_SDK_PATH}/include)
    link_directories(${VULKAN_SDK_PATH}/Lib)
    link_libraries(vulkan-1)

    # GLFW on Windows, without the help of pkg-config
    set(GLFW_PATH c:/GLFW_Libs)
    include_directories(${GLFW_PATH}/include)
    link_directories(${GLFW_PATH}/lib-vc2019)
    link_libraries(glfw3)
endif()

add_executable(${VULKAN_PROJECT_NAME} main.cpp)

# Manage shader compilation - seems overly complex.
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

# Look at some vars
message(CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR})
message(CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR})
message(CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME})
message(CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE})
foreach(file ${SPV_FILES})
    message(SPV_FILE: ${file})
endforeach()