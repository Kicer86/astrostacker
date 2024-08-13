
find_package(GTest REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)

find_program(Python python REQUIRED)

add_executable(astro-stacker-tests
    test_config.cpp
    test_utils.cpp
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
        spdlog::spdlog
)

add_test(
    NAME AstroStackerTests
    COMMAND astro-stacker-tests
)

add_test(
    NAME FunctionalTests
    COMMAND ${Python} functional_tests.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(FunctionalTests
    PROPERTIES
        ENVIRONMENT AS_PATH=$<TARGET_FILE:astro-stacker>
)
