
cmake_minimum_required(VERSION 3.28)
project(astro-stacker
  LANGUAGES CXX
)

if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 23)

set(CMAKE_FIND_PACKAGE_PREFER_CONFIG TRUE)

enable_testing()

add_subdirectory(tests)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(spdlog REQUIRED)

add_executable(astro-stacker
    main.cpp
)

target_sources(astro-stacker
  PUBLIC
    FILE_SET CXX_MODULES FILES
      aberration_fixer.cpp
      config.cpp
      execution_plan_builder.cpp
      file_manager.cpp
      frame_extractor.cpp
      ifile_manager.cpp
      image_extractor.cpp
      images_aligner.cpp
      images_cropper.cpp
      images_enhancer.cpp
      images_picker.cpp
      images_splitter.cpp
      images_stacker.cpp
      object_localizer.cpp
      transparency_applier.cpp
      utils.cpp
)

if (MSVC)
    target_compile_definitions(astro-stacker PRIVATE MSVC)
    target_compile_options(astro-stacker PRIVATE /openmp:llvm)
endif()

target_link_libraries(astro-stacker
    PRIVATE
        Boost::program_options
        opencv_tracking
        opencv_videoio
        opencv_photo
        OpenMP::OpenMP_CXX
        spdlog::spdlog
)
