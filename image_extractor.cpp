
module;

#include <algorithm>
#include <filesystem>
#include <ranges>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <libraw/libraw.h>
#include <opencv2/opencv.hpp>


export module image_extractor;
import utils;

namespace
{
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

    struct RawProcessorWrapper final
    {
        RawProcessorWrapper() {}
        ~RawProcessorWrapper()
        {
            rawProcessor.recycle();
        }

        LibRaw* operator->()
        {
            return &rawProcessor;
        }

    private:
        LibRaw rawProcessor;
    };


    template<typename T, typename V>
    constexpr T saturate_cast(const V& v)
    {
        const auto clamped = std::clamp(v, static_cast<V>(std::numeric_limits<T>::min()), static_cast<V>(std::numeric_limits<T>::max()));

        return static_cast<T>(clamped);
    }


    cv::Mat loadRawImage(const std::filesystem::path& path)
    {
        const auto pathStr = path.string();
        RawProcessorWrapper rawProcessor;

        if (rawProcessor->open_file(pathStr.c_str()) != LIBRAW_SUCCESS)
            throw std::runtime_error(std::format("Cannot open input file: {}", pathStr));

        if (rawProcessor->unpack() != LIBRAW_SUCCESS)
            throw std::runtime_error(std::format("Error when unpacking RAW file data: {}", pathStr));

        // Get raw Bayer image dimensions and the data pointer
        const int width = rawProcessor->imgdata.sizes.iwidth;
        const int height = rawProcessor->imgdata.sizes.iheight;
        const int black_level = rawProcessor->imgdata.color.black; // Black level value

        // Create OpenCV Mat to store the raw Bayer data (16-bit single-channel)
        cv::Mat raw_image(height, width, CV_16UC1, rawProcessor->imgdata.rawdata.raw_image);

        // ------------------- Step 1: Black Level Subtraction -------------------
        // Subtract the black level from each pixel
        for (int i = 0; i < raw_image.rows; ++i)
            for (int j = 0; j < raw_image.cols; ++j)
            {
                const auto v = raw_image.at<uint16_t>(i, j);
                const auto vb = v - black_level;
                raw_image.at<uint16_t>(i, j) = std::max(0, vb);
            }

        // ------------------- Step 2: Demosaicing (Bayer to RGB conversion) -------------------
        // Use OpenCV's cvtColor to convert Bayer-patterned raw data to full RGB image
        cv::Mat demosaiced_image;
        cv::cvtColor(raw_image, demosaiced_image, cv::COLOR_BayerBG2BGR); // Adjust for your camera's Bayer pattern

        // ------------------- Step 3: White Balance Correction -------------------
        // Get white balance multipliers from metadata
        const float wb_red_multiplier = rawProcessor->imgdata.color.cam_mul[0];
        const float wb_green_multiplier = rawProcessor->imgdata.color.cam_mul[1];
        const float wb_blue_multiplier = rawProcessor->imgdata.color.cam_mul[2];

        // Apply white balance correction
        for (int i = 0; i < demosaiced_image.rows; ++i)
            for (int j = 0; j < demosaiced_image.cols; ++j)
            {
                cv::Vec3b& pixel = demosaiced_image.at<cv::Vec3b>(i, j);
                pixel[0] = saturate_cast<uchar>(pixel[0] * wb_blue_multiplier);  // Blue channel
                pixel[1] = saturate_cast<uchar>(pixel[1] * wb_green_multiplier); // Green channel
                pixel[2] = saturate_cast<uchar>(pixel[2] * wb_red_multiplier);   // Red channel
            }

        // ------------------- Step 4: Gamma Correction -------------------
        // Convert image to floating point for gamma correction
        cv::Mat gamma_corrected_image;
        demosaiced_image.convertTo(gamma_corrected_image, CV_32F, 1.0 / 255.0);

        // Apply gamma correction (typical gamma is 2.2)
        cv::pow(gamma_corrected_image, 1.0 / 2.2, gamma_corrected_image);

        // Convert back to 8-bit
        gamma_corrected_image.convertTo(gamma_corrected_image, CV_8UC3, 255.0);

        // ------------------- Step 5: Sharpening (Unsharp Mask) -------------------
        cv::Mat sharpened_image;
        cv::GaussianBlur(gamma_corrected_image, sharpened_image, cv::Size(0, 0), 3);
        cv::addWeighted(gamma_corrected_image, 1.5, sharpened_image, -0.5, 0, sharpened_image);

        return sharpened_image;
    }
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
            image = loadRawImage(path);
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
