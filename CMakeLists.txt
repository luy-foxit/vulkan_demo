cmake_minimum_required(VERSION 3.5)
project(VulkanDemo LANGUAGES C CXX)

set(OS          "linux" CACHE STRING "Target OS: linux, anroid")

#set(CMAKE_BUILD_TYPE "Debug")
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Release")
    message("cmake compiler in release mode")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
else()
    message("cmake compiler in debug mode")
    set(CMAKE_CXX_FLAGS "-g -O0")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

####################################
## to use C++11
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
if (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
endif()
####################################

# add OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# add vulkan
find_package(Vulkan REQUIRED)
include_directories(${Vulkan_INCLUDE_DIR})

#glslangValidator
find_program(GLSLANGVALIDATOR_EXECUTABLE NAMES glslangValidator PATHS $ENV{VULKAN_SDK}/bin NO_CMAKE_FIND_ROOT_PATH)
message(STATUS "Found glslangValidator: ${GLSLANGVALIDATOR_EXECUTABLE}")

#compiler glsl
file(GLOB_RECURSE SHADER_SRCS "src/layer/shader/*.comp")
set(SHADER_HEX_FOLDER ${CMAKE_CURRENT_BINARY_DIR}/include)
file(MAKE_DIRECTORY ${SHADER_HEX_FOLDER})
foreach(SHADER_SRC ${SHADER_SRCS})
    get_filename_component(SHADER_SRC_NAME_WE ${SHADER_SRC} NAME_WE)
    set(SHADER_SPV_HEX_FILE ${SHADER_HEX_FOLDER}/${SHADER_SRC_NAME_WE}.spv.hex.h)

    add_custom_command(
        OUTPUT ${SHADER_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        #ARGS -V -s -e ${SHADER_SRC_NAME_WE} --source-entrypoint main -x -o ${SHADER_SPV_HEX_FILE} ${SHADER_SRC}
        ARGS -V ${SHADER_SRC} -x -o ${SHADER_SPV_HEX_FILE}
        DEPENDS ${SHADER_SRC}
        COMMENT "Building SPIR-V module ${SHADER_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)

    string(APPEND layer_shader_spv_data "static const uint32_t ${SHADER_SRC_NAME_WE}_spv_data[] = {\n#include                  \"${SHADER_SRC_NAME_WE}.spv.hex.h\"\n};\n")
    string(APPEND layer_shader_registry "{\"${SHADER_SRC_NAME_WE}\",${SHADER_SRC_NAME_WE}_spv_data,                            sizeof(${SHADER_SRC_NAME_WE}_spv_data)},\n")

    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_SPV_HEX_FILE})
endforeach()

configure_file(src/layer/shader/layer_shader_spv_data.h.in ${SHADER_HEX_FOLDER}/layer_shader_spv_data.h)
configure_file(src/layer/shader/layer_shader_registry.h.in ${SHADER_HEX_FOLDER}/layer_shader_registry.h)
include_directories(${SHADER_HEX_FOLDER})

file(GLOB srcs "src/*.cpp")
file(GLOB layer_srcs "src/layer/*.cpp")
ADD_EXECUTABLE(
    vulkan_demo
    ${srcs}
    ${layer_srcs}
    )

TARGET_LINK_LIBRARIES(
    vulkan_demo
    ${Vulkan_LIBRARY}
    ${OpenCV_LIBRARIES}
    )

add_custom_target(generate-spirv DEPENDS ${SHADER_SPV_HEX_FILES})
add_dependencies(vulkan_demo generate-spirv)

# print
function(string_status text value)
    set(status_placeholder_length 24)
    string(RANDOM LENGTH ${status_placeholder_length} ALPHABET " " status_placeholder)
    string(LENGTH "${text}" status_text_length)
    if(status_text_length LESS status_placeholder_length)
        string(SUBSTRING "${text}${status_placeholder}" 0 ${status_placeholder_length} status_text)
    else()
        set(status_text "${text}")
    endif()
    string(TOLOWER ${value} status_value)
    message(STATUS "${status_text}  : ${status_value}")
endfunction()
message("===============================================================================")
string_status("C compiler"                     ${CMAKE_C_COMPILER})
string_status("C++ compiler"                   ${CMAKE_CXX_COMPILER})
string_status("opencv libs"                    ${OpenCV_LIBRARIES})
string_status("opencv include"                 ${OpenCV_INCLUDE_DIRS})
string_status("vulkan libs"                    ${Vulkan_LIBRARY})
string_status("vulkan include"                 ${Vulkan_INCLUDE_DIR})
message("===============================================================================")



