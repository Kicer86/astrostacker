
module;

#include <algorithm>
#include <filesystem>
#include <numeric>
#include <variant>
#include <vector>
#include <ranges>
#include <string>
#include <opencv2/opencv.hpp>

export module images_picker;

import utils;


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

    std::vector<size_t> selectTop(const std::vector<std::pair<double, size_t>>& images, int percent = 50) {
        std::vector<size_t> top;
        std::ranges::transform(images, std::back_inserter(top), [](const std::pair<double, size_t> score) {return score.second;});

        const size_t elements = static_cast<size_t>(std::ceil(top.size() * percent / 100.0));

        return {top.begin(), top.begin() + elements};
    }
}

export struct MedianPicker {};
export using PickerMethod = std::variant<int, MedianPicker>;

export std::vector<std::filesystem::path> pickImages(const std::filesystem::path& dir, std::span<const std::filesystem::path> images, const PickerMethod& method)
{
    std::vector<std::pair<double, size_t>> score;

    const size_t count = images.size();
    score.resize(count);

    forEach(images, [&](const size_t i)
    {
        const cv::Mat image = cv::imread(images[i].string());
        const double s = computeSharpness(image);
        const double c = computeContrast(image);

        score[i] = {s * c, i};
    });

    auto cmp = [](const auto& lhs, const auto& rhs)
    {
        return lhs.first > rhs.first;
    };

    std::sort(score.begin(), score.end(), cmp);

    if (const auto medianMethod = std::get_if<MedianPicker>(&method))
    {
        const auto top = selectTop(score);
        const auto topImages = top | std::ranges::views::transform([&](const auto& idx) { return images[idx]; });
        const auto topPaths = createLinks(std::vector<std::filesystem::path>(topImages.begin(), topImages.end()), dir);

        return topPaths;
    }
    else if (const auto topMethod = std::get_if<int>(&method))
    {
        const auto top = selectTop(score, *topMethod);
        const auto topImages = top | std::ranges::views::transform([&](const auto& idx) { return images[idx]; });
        const auto topPaths = createLinks(std::vector<std::filesystem::path>(topImages.begin(), topImages.end()), dir);

        return topPaths;
    }
    else
        return {};
}
