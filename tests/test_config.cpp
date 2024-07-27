
#include <gtest/gtest.h>
#include <gmock/gmock.h>


#include <ranges>
#include <string>

import config;
import images_picker;

using testing::Contains;


TEST(ConfigTest, defaults)
{
    std::vector<std::string> argv_str = {"test.bin", "--working-dir", "somedir", "input_file.mp4"};
    auto argv_view = argv_str | std::views::transform([](std::string& str) { return str.data(); });
    std::vector<char *> argv(argv_view.begin(), argv_view.end());

    const auto config = readParams(argv.size(), &argv[0]);

    EXPECT_FALSE(config.crop.has_value());
    EXPECT_TRUE(config.doObjectDetection);
    EXPECT_THAT(config.inputFiles, Contains("input_file.mp4"));
    EXPECT_TRUE(std::holds_alternative<MedianPicker>(config.pickerMethod));
    EXPECT_EQ(config.skip, 0);
    EXPECT_FALSE(config.split.has_value());
    EXPECT_TRUE(config.wd.string().starts_with("somedir"));
}
