
module;

#include <filesystem>
#include <ranges>
#include <vector>

export module image_extractor;
import utils;


auto collectImages(const std::filesystem::path& inputDir)
{
    auto files =
        std::filesystem::directory_iterator(inputDir) |
        std::views::filter([](const std::filesystem::directory_entry &entry)
        {
            return entry.is_regular_file();
        }) |
        std::views::transform([](const std::filesystem::directory_entry &entry)
        {
            return entry.path();
        });

    return files | std::ranges::to<std::vector>();
}

export size_t countImages(const std::filesystem::path& inputDir)
{
    return collectImages(inputDir).size();
}

export std::vector<std::filesystem::path> collectImages(const std::filesystem::path& dir, std::span<const std::filesystem::path> files, size_t firstFrame, size_t lastFrame)
{
    if (files.size() != 1)
        throw std::runtime_error("Unexpected number of input elements: " + std::to_string(files.size()));

    const auto& input = files.front();

    if (not std::filesystem::is_directory(input))
        throw std::runtime_error("Input path: " + input.string() + " is not a directory");

    const auto images = collectImages(input);
    const auto count = images.size();

    if (count < lastFrame)
        throw std::out_of_range("last frame > number of frames");

    return Utils::processImages(std::ranges::subrange(images.begin() + firstFrame, images.begin() + lastFrame), dir, [](const auto& image)
    {
        return image;
    });
}
