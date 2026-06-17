cmake_minimum_required(VERSION 3.18)
project(mnist_ort_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime root dir")

if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "Set -DONNXRUNTIME_ROOT=<path-to-onnxruntime>")
endif()

add_executable(infer infer.cpp)

target_include_directories(infer PRIVATE "${ONNXRUNTIME_ROOT}/include")
target_link_directories(infer PRIVATE "${ONNXRUNTIME_ROOT}/lib")
target_link_libraries(infer PRIVATE onnxruntime)

set_target_properties(infer PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
    INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib")