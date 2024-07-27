
module;

#include <filesystem>
#include <vector>

#include <opencv2/opencv.hpp>

export module images_cropper;

import utils;


export std::vector<std::filesystem::path> cropImages(const std::filesystem::path& wd, std::span<const std::filesystem::path> images, const std::pair<int, int>& crop)
{
    const std::vector<std::filesystem::path> croppedImages = processImages(images, wd, [&crop](const cv::Mat& image)
    {
        const int height = image.rows;
        const int width = image.cols;

        const int cropWidth = std::min(crop.first, width);
        const int cropHeight = std::min(crop.second, height);

        const int centerX = width / 2;
        const int centerY = height / 2;

        const int startX = centerX - cropWidth / 2;
        const int startY = centerY - cropHeight / 2;

        const cv::Rect roi(startX, startY, cropWidth, cropHeight);
        const cv::Mat croppedImage = image(roi);

        return croppedImage;
    });

    return croppedImages;
}
