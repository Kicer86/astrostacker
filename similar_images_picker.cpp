
module;

#include <filesystem>
#include <ranges>
#include <span>

#include <opencv2/opencv.hpp>
#include <opencv2/quality.hpp>

export module similar_images_picker;


double computeSSIM(const cv::Mat& img1, const cv::Mat& img2)
{
    return cv::quality::QualitySSIM::compute(img1, img2, cv::noArray())[0];
}

cv::Mat similarityToDistanceMatrix(const cv::Mat& similarityMatrix)
{
    cv::Mat distanceMatrix = cv::Mat::zeros(similarityMatrix.size(), CV_64F);

    for (int i = 0; i < similarityMatrix.rows; ++i)
        for (int j = 0; j < similarityMatrix.cols; ++j)
            distanceMatrix.at<double>(i, j) = 1.0 - similarityMatrix.at<double>(i, j);

    return distanceMatrix;
}


export std::vector<std::filesystem::path> pickSimilarImages(const std::span<const std::filesystem::path> images, const std::filesystem::path& dir)
{
    const auto grayImagesRange = images | std::views::transform([](const std::filesystem::path& imagePath)
    {
        const cv::Mat image = cv::imread(imagePath);
        cv::Mat gray;
        cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        return gray;
    });

    const std::vector<cv::Mat> grayImages(grayImagesRange.begin(), grayImagesRange.end());

    const auto imagesCount = images.size();
    cv::Mat similarityMatrix = cv::Mat::zeros(imagesCount, imagesCount, CV_64F);

    //#pragma omp parallel for
    for (size_t i = 0; i < grayImages.size(); ++i)
        for (size_t j = i + 1; j < grayImages.size(); ++j)
        {
            const double ssim = computeSSIM(grayImages[i], grayImages[j]);
            similarityMatrix.at<double>(i, j) = ssim;
            similarityMatrix.at<double>(j, i) = ssim;
        }

    std::vector<double> similarities(similarityMatrix.begin<double>(), similarityMatrix.end<double>());
    std::ranges::sort(similarities);
    const double limit = similarities[similarities.size() * 7 / 8];

    std::vector<int> scores(imagesCount);
    for (int i = 0; i < imagesCount; i++)
        for (int j = 0; j < imagesCount; j++)
            scores[i] += similarityMatrix.at<double>(i, j) > limit;

    const auto best = std::ranges::max_element(scores);
    const auto pos = std::distance(scores.begin(), best);

    std::vector<std::filesystem::path> similarImages;
    for(int i = 0; i < imagesCount; i++)
        if (similarityMatrix.at<double>(pos, i) > limit)
        {
            const auto newFile = dir / std::format("{}.tiff", i);
            const auto targetPath = std::filesystem::relative(images[i], dir);
            std::filesystem::create_symlink(targetPath, newFile);
            similarImages.push_back(newFile);
        }

    return similarImages;
}

