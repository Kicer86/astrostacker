
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

    std::vector<size_t> selectTop(const std::vector<std::pair<double, size_t>>& images, int percent = 50)
    {
        std::vector<size_t> top;
        std::ranges::transform(images, std::back_inserter(top), [](const std::pair<double, size_t> score) {return score.second;});

        const size_t elements = static_cast<size_t>(std::ceil(top.size() * percent / 100.0));

        return {top.begin(), top.begin() + elements};
    }

    std::vector<size_t> selectBestWithFriends(const std::vector<std::pair<double, size_t>>& images, int friends)
    {
        std::vector<size_t> top;
        top.resize(friends * 2 + 1);

        // fill 'top' with indexes from bestIdx - friends to bestIdx + friends
        const auto& best = images.front();
        const auto bestIdx = best.second;
        std::iota(top.begin(), top.end(), bestIdx - friends);

        // eliminate elements out of scope
        const auto maxIdx = images.size();
        std::erase_if(top, [maxIdx](const std::size_t& idx) { return idx < 0 || idx >= maxIdx; });

        return top;
    }
}

export struct MedianPicker {};
export struct BestWithFriends { const int friends; };
export using PickerMethod = std::variant<int, MedianPicker, BestWithFriends>;

export std::vector<std::filesystem::path> pickImages(const std::filesystem::path& dir, std::span<const std::filesystem::path> images, const PickerMethod& method)
{
    std::vector<std::pair<double, size_t>> score;

    const size_t count = images.size();
    score.resize(count);

    Utils::forEach(images, [&](const size_t i)
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

    auto processTop = [&](std::span<const size_t> top)
    {
        const auto topImages = top | std::ranges::views::transform([&](const auto& idx) { return images[idx]; });
        const auto topPaths = Utils::copyFiles(std::vector<std::filesystem::path>(topImages.begin(), topImages.end()), dir);

        return topPaths;
    };

    if (const auto medianMethod = std::get_if<MedianPicker>(&method))
    {
        const auto top = selectTop(score);
        return processTop(top);
    }
    else if (const auto bestWithFriends = std::get_if<BestWithFriends>(&method))
    {
        const auto top = selectBestWithFriends(score, bestWithFriends->friends);
        return processTop(top);
    }
    else if (const auto topMethod = std::get_if<int>(&method))
    {
        const auto top = selectTop(score, *topMethod);
        return processTop(top);
    }
    else
        return {};
}
