
cmake_minimum_required(VERSION 3.28)
project(astro-stacker)

if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)

add_executable(astro-stacker
    main.cpp
)

target_sources(astro-stacker
  PUBLIC
    FILE_SET CXX_MODULES FILES
      frame_extractor.cpp
)

target_link_libraries(astro-stacker
    PRIVATE
        opencv_videoio
)
