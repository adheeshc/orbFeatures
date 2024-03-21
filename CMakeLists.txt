cmake_minimum_required(VERSION 3.0.0)
project(ORBFeatureMatcher)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN_INCLUDE_DIR})

set(source_files
    src/main.cpp
    src/ORBFeatureMatcherCV.cpp
    src/ORBFeatureMatcherCV.h
    src/CustomORBFeatureMatcher.h
    src/CustomORBFeatureMatcher.cpp
)

add_executable(ORBFeatureMatcher ${source_files})

target_link_libraries(ORBFeatureMatcher ${OpenCV_LIBS})
target_link_libraries(ORBFeatureMatcher Eigen3::Eigen)

# Compiler flags
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Enable SSE4.2 instruction set
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
endif()

# Suppress Warnings
# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#     # Suppress specific warnings for GCC and Clang
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-unused-parameter")
# elseif(MSVC)
#     # Suppress specific warnings for MSVC
#     add_compile_options(/wd4244 /wd4267)
# endif()