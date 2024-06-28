
module;

#include <concepts>
#include <filesystem>
#include <opencv2/opencv.hpp>

export module utils;


export template<typename T>
requires std::invocable<T, const cv::Mat &>
std::vector<std::filesystem::path> processImages(const std::vector<std::filesystem::path>& images, const std::filesystem::path& dir, T&& op)
{
    const auto imagesCount = images.size();
    std::vector<std::filesystem::path> resultPaths(imagesCount);

    #pragma omp parallel for
    for(size_t i = 0; i < imagesCount; i++)
    {
        const cv::Mat image = cv::imread(images[i]);
        const auto result = op(image);

        const auto path = dir / std::format("{}.tiff", i);
        cv::imwrite(path, result);

        resultPaths[i] = path;
    }

    return resultPaths;
}
