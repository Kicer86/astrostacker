
#include <gtest/gtest.h>
#include <gmock/gmock.h>


#include <ranges>
#include <string>

import config;
import images_picker;
import utils;

using testing::Contains;


TEST(ConfigTest, defaults)
{
    std::vector<std::string> argv_str = {"test.bin", "--working-dir", "somedir", "input_file.mp4"};
    auto argv_view = argv_str | std::views::transform([](std::string& str) { return str.data(); });
    std::vector<char *> argv(argv_view.begin(), argv_view.end());

    const auto config = Config::readParams(argv.size(), &argv[0]);

    EXPECT_FALSE(config.crop.has_value());
    EXPECT_TRUE(config.doObjectDetection);
    EXPECT_THAT(config.inputFiles, Contains("input_file.mp4"));
    EXPECT_TRUE(std::holds_alternative<MedianPicker>(config.pickerMethod));
    EXPECT_EQ(config.skip, 0);
    EXPECT_FALSE(config.split.has_value());
    EXPECT_TRUE(config.wd.string().starts_with("somedir"));
}

using CropParam = std::tuple<std::string_view, int, int, int, int>;

class CropParserTest: public testing::TestWithParam<CropParam> { };

INSTANTIATE_TEST_SUITE_P(
    ValidCrop, CropParserTest,
    testing::Values(
        CropParam{ "100x200", 100, 200, 0, 0 },
        CropParam{ "0x0,-5,-6", 0, 0, -5, -6 },
        CropParam{ "1000x550,17,13", 1000, 550, 17, 13 }
    )
);


TEST_P(CropParserTest, parsing)
{
    const auto& [input, w, h, x, y] = GetParam();

    const auto result = Utils::readCrop(input);
    ASSERT_TRUE(result);
    EXPECT_EQ(*result, (std::tuple{w, h, x, y}));
}
