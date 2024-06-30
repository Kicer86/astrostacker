
module;

#include <algorithm>
#include <filesystem>
#include <vector>
#include <opencv2/opencv.hpp>

export module images_stacker;


namespace
{
    cv::Mat averageStacking(const std::vector<std::filesystem::path>& images)
    {
        const cv::Mat firstImage = cv::imread(images.front());

        cv::Mat cumulative = cv::Mat::zeros(firstImage.size(), CV_64FC3);

        for (const auto& imagePath: images)
        {
            const cv::Mat image = cv::imread(imagePath);

            cv::Mat imageFloat;
            image.convertTo(imageFloat, CV_64FC3);

            cumulative += imageFloat;
        }

        cumulative /= static_cast<double>(images.size());

        cv::Mat result;
        cumulative.convertTo(result, firstImage.type());

        return result;
    }


    cv::Mat medianStacking(const std::vector<std::filesystem::path>& images)
    {
        // TODO: rewrite with std::mdspan
        const cv::Mat firstImage = cv::imread(images.front());
        const auto imagesCount = images.size();
        std::vector<cv::Vec3b> pixels(imagesCount * firstImage.rows * firstImage.cols);

        // Collect pixel values
        #pragma omp parallel for
        for (std::size_t i = 0; i < imagesCount; i++)
        {
            const cv::Mat image = cv::imread(images[i]);
            for (int y = 0; y < image.rows; ++y)
                for (int x = 0; x < image.cols; ++x)
                    pixels[y * image.cols * imagesCount + x * imagesCount + i] = image.at<cv::Vec3b>(y, x);
        }

        cv::Mat result(firstImage.size(), firstImage.type());

        // Compute the median for each pixel
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < result.rows; ++y)
            for (int x = 0; x < result.cols; ++x)
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


export std::vector<std::filesystem::path> stackImages(const std::vector<std::filesystem::path>& images, const std::filesystem::path& dir)
{
    const auto averageImg = averageStacking(images);

    const auto pathAvg = dir / "average.tiff";
    cv::imwrite(pathAvg, averageImg);

    const auto medianImg = medianStacking(images);

    const auto pathMdn = dir / "median.tiff";
    cv::imwrite(pathMdn, medianImg);

    return {pathAvg, pathMdn};
}
