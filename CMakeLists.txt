cmake_minimum_required(VERSION 3.16)
project(vk_compute_pathtracer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(ZLIB REQUIRED)
find_package(glfw3 3.3 QUIET)

add_executable(rt src/main.cpp)
target_link_libraries(rt PRIVATE Vulkan::Vulkan ZLIB::ZLIB)
if (TARGET glfw)
    target_link_libraries(rt PRIVATE glfw)
else()
    include_directories(/opt/homebrew/include /usr/local/include /usr/include)
    find_library(GLFW_LIB glfw HINTS /opt/homebrew/lib /usr/local/lib)
    target_link_libraries(rt PRIVATE ${GLFW_LIB})
endif()

# compile compute shaders with glslang
find_program(GLSLANG_VALIDATOR NAMES glslangValidator glslang REQUIRED)
set(SHADER_DIR_OUT ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_DIR_OUT})
set(SPIRV_OUTS)
foreach(SH pathtrace denoise finalize)
    set(SPIRV ${SHADER_DIR_OUT}/${SH}.comp.spv)
    add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.1 -o ${SPIRV}
                ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SH}.comp
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SH}.comp
        COMMENT "glslang: ${SH}.comp -> spv"
        VERBATIM)
    list(APPEND SPIRV_OUTS ${SPIRV})
endforeach()
add_custom_target(shaders ALL DEPENDS ${SPIRV_OUTS})
add_dependencies(rt shaders)
target_compile_definitions(rt PRIVATE
    SHADER_DIR="${SHADER_DIR_OUT}/"
    ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets/")
