
find_package(GTest)

add_executable(astro-stacker-tests
    test_basic.cpp
)

target_link_libraries(astro-stacker-tests
    PRIVATE
        GTest::gtest_main
)
