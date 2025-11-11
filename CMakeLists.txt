cmake_minimum_required(VERSION 3.2)

project(IluvatarPipeLine LANGUAGES CXX CUDA)


# 设置C++标准和调试选项
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_COMPILER "clang++" CACHE STRING "CUDA compiler.")


include_directories(/usr/local/corex/include)
link_directories(/usr/local/corex/lib64)


# 包含FFmpeg库
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale libswresample)


# 包含头文件目录
include_directories(inc)

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

option(ENABLE_YOLOV5_VALIDATION "Enable YoloV5 label/score validation" OFF)
if(ENABLE_YOLOV5_VALIDATION)
    add_definitions(-DYOLOV5_VALIDATION_ENABLED)
endif()

file(GLOB SRC_FILES src/*.cpp src/*/*.cpp  src/*/*/*.cpp src/*/*/*.cu)

add_library(ix_pipeline_help SHARED ${SRC_FILES})

target_link_libraries(ix_pipeline_help ${FFMPEG_LIBRARIES} spdlog fmt ixrt ixrt_plugin ixrtonnxparser nvjpeg nvcuvid cuda cudart)

add_subdirectory(test)