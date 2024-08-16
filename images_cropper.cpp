
module;

#include <filesystem>
#include <vector>

#include <opencv2/opencv.hpp>

export module images_cropper;

import utils;


export std::vector<std::filesystem::path> cropImages(const std::filesystem::path& wd, std::span<const std::filesystem::path> images, const std::tuple<int, int, int, int>& crop)
{
    const std::vector<std::filesystem::path> croppedImages = Utils::processImages(images, wd, [&crop](const cv::Mat& image)
    {
        const int height = image.rows;
        const int width = image.cols;

        const int cropWidth = std::min(std::get<0>(crop), width);
        const int cropHeight = std::min(std::get<1>(crop), height);
        const int cropDX = std::get<2>(crop);
        const int cropDY = std::get<3>(crop);

        const int centerX = width / 2;
        const int centerY = height / 2;

        const int startX = centerX + cropDX - cropWidth / 2;
        const int startY = centerY + cropDY - cropHeight / 2;

        const cv::Rect roi(startX, startY, cropWidth, cropHeight);
        const cv::Mat croppedImage = image(roi);

        return croppedImage;
    });

    return croppedImages;
}
