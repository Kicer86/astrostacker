
module;

#include <algorithm>
#include <filesystem>
#include <span>
#include <vector>
#include <opencv2/opencv.hpp>

export module images_stacker;


namespace
{
    cv::Mat averageStacking(const std::span<const std::filesystem::path> images)
    {
        const cv::Mat firstImage = cv::imread(images.front().string());

        cv::Mat cumulative = cv::Mat::zeros(firstImage.size(), CV_64FC3);

        for (const auto& imagePath: images)
        {
            const cv::Mat image = cv::imread(imagePath.string());

            cv::Mat imageFloat;
            image.convertTo(imageFloat, CV_64FC3);

            cumulative += imageFloat;
        }

        cumulative /= static_cast<double>(images.size());

        cv::Mat result;
        cumulative.convertTo(result, firstImage.type());

        return result;
    }


    cv::Mat medianStacking(const std::span<const std::filesystem::path> images)
    {
        // TODO: rewrite with std::mdspan
        const cv::Mat firstImage = cv::imread(images.front().string());
        const auto imagesCount = images.size();
        std::vector<cv::Vec3b> pixels(imagesCount * firstImage.rows * firstImage.cols);

        // Collect pixel values
        #pragma omp parallel for
        for (size_t i = 0; i < imagesCount; i++)
        {
            const cv::Mat image = cv::imread(images[i].string());
            for (int y = 0; y < image.rows; ++y)
                for (int x = 0; x < image.cols; ++x)
                    pixels[y * image.cols * imagesCount + x * imagesCount + i] = image.at<cv::Vec3b>(y, x);
        }

        cv::Mat result(firstImage.size(), firstImage.type());

        // Compute the median for each pixel
        #pragma omp parallel for                                // TODO: restore collapse(2)
        for (size_t y = 0; y < result.rows; y++)
            for (size_t x = 0; x < result.cols; x++)
            {
                const std::span<cv::Vec3b> px(&pixels[y * result.cols * imagesCount + x * imagesCount], imagesCount);
                std::sort(px.begin(), px.end(), [](const cv::Vec3b& a, const cv::Vec3b& b)
                {
                    return cv::norm(a) < cv::norm(b);
                });
                result.at<cv::Vec3b>(y, x) = px[px.size() / 2];
            }

        return result;
    }
}


export std::vector<std::filesystem::path> stackImages(const std::filesystem::path& dir, std::span<const std::filesystem::path> images)
{
    const auto averageImg = averageStacking(images);

    const auto pathAvg = dir / "average.png";
    cv::imwrite(pathAvg.string(), averageImg);

    const auto medianImg = medianStacking(images);

    const auto pathMdn = dir / "median.png";
    cv::imwrite(pathMdn.string(), medianImg);

    return {pathAvg, pathMdn};
}
