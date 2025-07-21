cmake_minimum_required(VERSION 3.18)
project(fused_mm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Find or fetch Eigen3
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Eigen)
    set(EIGEN3_INCLUDE_DIR ${eigen_SOURCE_DIR})
else()
    set(EIGEN3_INCLUDE_DIR ${EIGEN3_INCLUDE_DIR})
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${EIGEN3_INCLUDE_DIR})

# Add your C++ source files
file(GLOB_RECURSE CPP_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/regularization/ctvlib.cpp"
)

# Create the pybind11 module using your existing bindings
pybind11_add_module(ctvlib 
    regularization/bindings.cpp
    ${CPP_SOURCES}
)

# Set properties for the compiled module
target_compile_definitions(ctvlib PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_include_directories(ctvlib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/regularization)
target_include_directories(ctvlib PRIVATE ${EIGEN3_INCLUDE_DIR})

# Link Eigen3 if found as a target
if(TARGET Eigen3::Eigen)
    target_link_libraries(ctvlib PRIVATE Eigen3::Eigen)
endif()

# Platform-specific settings
if(MSVC)
    target_compile_options(ctvlib PRIVATE /bigobj)
endif()