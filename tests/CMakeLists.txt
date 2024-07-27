
find_package(GTest)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(OpenCV REQUIRED)

add_executable(astro-stacker-tests
    test_config.cpp
)


target_sources(astro-stacker-tests
  PUBLIC
    FILE_SET CXX_MODULES
    BASE_DIRS
        ${PROJECT_SOURCE_DIR}
    FILES
        ${PROJECT_SOURCE_DIR}/config.cpp
        ${PROJECT_SOURCE_DIR}/images_picker.cpp
        ${PROJECT_SOURCE_DIR}/utils.cpp
)


target_link_libraries(astro-stacker-tests
    PRIVATE
        GTest::gtest_main
        Boost::program_options
        opencv_tracking
)

add_test(
    NAME AstroStackerTests
    COMMAND astro-stacker-tests
)
