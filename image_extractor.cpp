
module;

#include <filesystem>
#include <ranges>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <libraw/libraw.h>
#include <opencv2/opencv.hpp>


export module image_extractor;
import utils;


auto collectImages(const std::filesystem::path& inputDir)
{
    auto files =
        std::filesystem::recursive_directory_iterator(inputDir) |
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

    std::vector<std::filesystem::path> result;
    result.resize(lastFrame - firstFrame);

    Utils::forEach(std::ranges::subrange(images.begin() + firstFrame, images.begin() + lastFrame), [&](const auto& i)
    {
        const auto& path = images[i];
        auto extension = path.extension().string();
        boost::algorithm::to_lower(extension);

        cv::Mat image;
        const auto pathStr = path.string();

        if (extension == ".nef")
        {
            LibRaw rawProcessor;

            if (rawProcessor.open_file(pathStr.c_str()) != LIBRAW_SUCCESS)
                throw std::runtime_error(std::format("Cannot open input file: {}", pathStr));

            if (rawProcessor.unpack() != LIBRAW_SUCCESS)
                throw std::runtime_error(std::format("Error when unpacking RAW file data: {}", pathStr));

            rawProcessor.dcraw_process();
            libraw_processed_image_t* rawImage = rawProcessor.dcraw_make_mem_image();
            if (!rawImage)
                throw std::runtime_error(std::format("No RAW image: {}", pathStr));

            cv::Mat matImage(rawImage->height, rawImage->width, CV_8UC3, rawImage->data);
            image = matImage.clone();

            LibRaw::dcraw_clear_mem(rawImage);
            rawProcessor.recycle();
        }
        else
            image = cv::imread(pathStr);

        const auto filename = path.stem();
        const auto newLocation = dir / (filename.string() + ".png");

        cv::imwrite(newLocation.string(), image);

        result[i] = newLocation;
    });

    return result;
}
