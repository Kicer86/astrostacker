
#include <gtest/gtest.h>
#include <gmock/gmock.h>



import utils;

struct SplitParam
{
    size_t first;
    size_t last;
    size_t groups;

    std::vector<std::pair<size_t, size_t>> result;
};

class SplitTest: public testing::TestWithParam<SplitParam> { };

INSTANTIATE_TEST_SUITE_P(
    EqualSplit, SplitTest,
    testing::Values(
        SplitParam{0, 10, 2, {std::pair<size_t, size_t>{0, 5}, std::pair<size_t, size_t>{5, 10}}},
        SplitParam{3, 13, 2, {std::pair<size_t, size_t>{3, 8}, std::pair<size_t, size_t>{8, 13}}},
        SplitParam{100, 200, 4, {std::pair<size_t, size_t>{100, 125},
                                 std::pair<size_t, size_t>{125, 150},
                                 std::pair<size_t, size_t>{150, 175},
                                 std::pair<size_t, size_t>{175, 200}}}
    )
);


TEST_P(SplitTest, split)
{
    const auto& [first, last, groups, expectedResult] = GetParam();
    const auto results = split(std::pair{first, last}, groups);

    EXPECT_EQ(results, expectedResult);
}
