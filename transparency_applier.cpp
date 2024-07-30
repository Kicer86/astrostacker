
module;

#include <filesystem>
#include <span>
#include <vector>

#include <opencv2/opencv.hpp>


export module transparency_applier;
import utils;


export std::vector<std::filesystem::path> applyTransparency(const std::filesystem::path& dir, const std::span<const std::filesystem::path> images, int threshold)
{
    const std::vector<std::filesystem::path> transparent = processImages(images, dir, [&threshold](const cv::Mat& image)
    {
        cv::Mat rgbaImage;
        cv::cvtColor(image, rgbaImage, cv::COLOR_BGR2BGRA);

        for (int y = 0; y < rgbaImage.rows; ++y)
            for (int x = 0; x < rgbaImage.cols; ++x)
            {
                cv::Vec4b& pixel = rgbaImage.at<cv::Vec4b>(y, x);
                if (pixel[0] <= threshold && pixel[1] <= threshold && pixel[2] <= threshold)
                    pixel[3] = 0;
            }

        return rgbaImage;
    });

    return transparent;
}
