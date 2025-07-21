cmake_minimum_required(VERSION 3.15)
project(fused_mm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Include directories
include_directories(src/fused_mm/cpp/include)

# Your C++ source files
file(GLOB_RECURSE SOURCES 
    "fused_mm/cpp/src/*.cpp" 
    "fused_mm/cpp/src/*.hpp"
    "fused_mm/cpp/include/*.hpp"
)

# Create the pybind11 module
pybind11_add_module(_core ${SOURCES})

# Important: set the output name and properties
set_target_properties(_core PROPERTIES
    OUTPUT_NAME "_core"
    CXX_VISIBILITY_PRESET hidden
)

# Link libraries (adjust as needed)
target_link_libraries(_core PRIVATE 
    # Add your dependencies here (OpenMP, BLAS, etc.)
)

# Install the module into the Python package
install(TARGETS _core DESTINATION fused_mm)