
module;

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

export module frames_picker;


namespace
{
    double computeSharpness(const cv::Mat& img) {
        cv::Mat gray, laplacian;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::Laplacian(gray, laplacian, CV_64F);
        cv::Scalar mu, sigma;
        cv::meanStdDev(laplacian, mu, sigma);
        return sigma.val[0] * sigma.val[0];
    }

    double computeContrast(const cv::Mat& img) {
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::Scalar mu, sigma;
        cv::meanStdDev(gray, mu, sigma);
        return sigma.val[0];
    }
}


export std::vector<std::string> pickImages(const std::vector<std::string>& images)
{
    std::vector<std::string> picks;
    std::vector<std::pair<double, int>> sharpness;
    std::vector<std::pair<double, int>> contrast;
    std::vector<std::pair<double, int>> score;

    const int count = images.size();
    sharpness.resize(count);
    contrast.resize(count);
    score.resize(count);

    #pragma omp parallel for
    for(int i = 0; i < count; i++)
    {
        const cv::Mat image = cv::imread(images[i]);
        const double s = computeSharpness(image);
        const double c = computeContrast(image);

        sharpness[i] = {s, i};
        contrast[i] = {c, i};
        score[i] = {s * c, i};
    }

    auto cmp = [](const auto& lhs, const auto& rhs)
    {
        return lhs.first > rhs.first;
    };

    std::sort(sharpness.begin(), sharpness.end(), cmp);
    std::sort(contrast.begin(), contrast.end(), cmp);
    std::sort(score.begin(), score.end(), cmp);

    std::cout << "Sharpest images:\n";
    for(int i = 0; i < std::min(10, count); i++)
        std::cout << images[sharpness[i].second] << " with sharpness: " << sharpness[i].first << "\n";
    std::cout << "\n";

    std::cout << "Images with biggest contrast:\n";
    for(int i = 0; i < std::min(10, count); i++)
        std::cout << images[contrast[i].second] << " with contrast: " << contrast[i].first << "\n";
    std::cout << "\n";

    std::cout << "Overall score:\n";
    for(int i = 0; i < std::min(10, count); i++)
    {
        std::cout << images[score[i].second] << " with score: " << score[i].first << "\n";
        picks.push_back(images[score[i].second]);
    }

    return picks;
}
