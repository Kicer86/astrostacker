
module;

#include <algorithm>
#include <filesystem>
#include <numeric>
#include <vector>
#include <ranges>
#include <string>
#include <opencv2/opencv.hpp>

export module images_picker;


namespace
{
    double computeSharpness(const cv::Mat& img)
    {
        cv::Mat gray, laplacian;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::Laplacian(gray, laplacian, CV_64F);
        cv::Scalar mu, sigma;
        cv::meanStdDev(laplacian, mu, sigma);
        return sigma.val[0] * sigma.val[0];
    }

    double computeContrast(const cv::Mat& img)
    {
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::Scalar mu, sigma;
        cv::meanStdDev(gray, mu, sigma);
        return sigma.val[0];
    }

    double calculateMean(const std::vector<double>& data)
    {
        const double sum = std::accumulate(data.begin(), data.end(), 0.0);
        return sum / static_cast<double>(data.size());
    }

    double calculateStdDev(const std::vector<double>& data, double mean)
    {
        const double sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
        return std::sqrt(sq_sum / static_cast<double>(data.size()) - mean * mean);
    }

    std::vector<size_t> selectTopImagesZScores(const std::vector<std::pair<double, size_t>>& images, double threshold = 1.0) {
        std::vector<double> scores;
        std::ranges::transform(images, std::back_inserter(scores), [](const std::pair<double, size_t> score) {return score.first;});

        double mean = calculateMean(scores);
        double stdDev = calculateStdDev(scores, mean);
        std::vector<size_t> topImages;

        for (const auto& image: images)
        {
            const double zScore = (image.first - mean) / stdDev;
            if (zScore > threshold)
                topImages.push_back(image.second);
        }

        return topImages;
    }
}


export std::vector<std::filesystem::path> pickImages(const std::vector<std::filesystem::path>& images)
{
    std::vector<std::pair<double, size_t>> score;

    const size_t count = images.size();
    score.resize(count);

    #pragma omp parallel for
    for(size_t i = 0; i < count; i++)
    {
        const cv::Mat image = cv::imread(images[i]);
        const double s = computeSharpness(image);
        const double c = computeContrast(image);

        score[i] = {s * c, i};
    }

    auto cmp = [](const auto& lhs, const auto& rhs)
    {
        return lhs.first > rhs.first;
    };

    std::sort(score.begin(), score.end(), cmp);

    const auto top = selectTopImagesZScores(score);
    const auto topImages = top | std::ranges::views::transform([&](const auto& idx) { return images[idx]; });

    return {topImages.begin(), topImages.end()};
}
