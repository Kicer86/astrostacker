
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


INSTANTIATE_TEST_SUITE_P(
    EdgeCases, SplitTest,
    testing::Values(
        // Unevent split
        SplitParam{18, 28, 3, {std::pair<size_t, size_t>{18, 22},
                               std::pair<size_t, size_t>{22, 26},
                               std::pair<size_t, size_t>{26, 28}}},
        // Not enought elements for all groups
        SplitParam{11, 15, 13, {std::pair<size_t, size_t>{11, 12},
                                std::pair<size_t, size_t>{12, 13},
                                std::pair<size_t, size_t>{13, 14},
                                std::pair<size_t, size_t>{14, 15}}}
    )
);


TEST_P(SplitTest, split)
{
    const auto& [first, last, groups, expectedResult] = GetParam();
    const auto results = Utils::split(std::pair{first, last}, groups);

    EXPECT_EQ(results, expectedResult);
}
