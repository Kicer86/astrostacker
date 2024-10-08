
module;

#include <filesystem>
#include <format>
#include <limits>
#include <vector>
#include <opencv2/opencv.hpp>

export module images_aligner;


namespace
{
    cv::Rect calculateCrop(const cv::Rect& imageSize, const std::vector<cv::Mat>& transformations)
    {
        cv::Rect2f cropSum = imageSize;

        for(const auto& transformation: transformations)
        {
            cv::Rect2f cropped = imageSize;

            if (transformation.at<float>(0, 2) < 0)
            {
                cropped.x = transformation.at<float>(0, 2) * -1;
                cropped.width -= cropped.x;
            }

            if (transformation.at<float>(0, 2) > 0)
                cropped.width -= transformation.at<float>(0, 2);

            if (transformation.at<float>(1, 2) < 0)
            {
                cropped.y = transformation.at<float>(1, 2) * -1;
                cropped.height -= cropped.y;
            }

            if (transformation.at<float>(1, 2) > 0)
                cropped.height -= transformation.at<float>(1, 2);

            cropSum &= cropped;
        }

        cropSum.x = std::ceil(cropSum.x);
        cropSum.y = std::ceil(cropSum.y);
        cropSum.width = std::floor(cropSum.width);
        cropSum.height = std::floor(cropSum.height);

        return cropSum;
    }

    auto findTransformation(const cv::Mat& referenceImageGray, const cv::Mat& imageGray)
    {
        const int number_of_iterations = 5000;
        const double termination_eps = 5e-5;
        const cv::TermCriteria criteria (cv::TermCriteria::COUNT + cv::TermCriteria::EPS, number_of_iterations, termination_eps);

        cv::Mat warp_matrix = cv::Mat::eye(3, 3, CV_32F);
        cv::findTransformECC(referenceImageGray, imageGray, warp_matrix, cv::MOTION_HOMOGRAPHY, criteria);

        return warp_matrix;
    }

    std::pair<std::vector<cv::Mat>, cv::Size> calculateTransformations(const std::span<const std::filesystem::path> images)
    {
        const auto& first = images.front();
        const auto referenceImage = cv::imread(first.string());
        cv::Size minimalSize = referenceImage.size();

        cv::Mat referenceImageGray;
        cv::cvtColor(referenceImage, referenceImageGray, cv::COLOR_RGB2GRAY);

        const cv::Rect firstImageSize(0, 0, referenceImage.size().width, referenceImage.size().height);

        // calculate required transformations
        std::vector<cv::Mat> transformations(1, []{
            cv::Mat mat = cv::Mat::eye(3, 3, CV_32F);
            mat.at<float>(0, 0) = 1.;
            mat.at<float>(1, 1) = 1.;

            return mat;
        }());  // insert empty transformation matrix for first image

        const auto imagesCount = images.size();
        transformations.resize(imagesCount);

        #pragma omp parallel for
        for (size_t i = 1; i < imagesCount; i++)
        {
            const auto& next = images[i];
            const auto image = cv::imread(next.string());

            cv::Mat imageGray;
            cv::cvtColor(image, imageGray, cv::COLOR_RGB2GRAY);

            const auto transformation = findTransformation(referenceImageGray, imageGray);

            transformations[i] = transformation;

            #pragma omp critical
            {
                minimalSize.width = std::min(minimalSize.width, image.size().width);
                minimalSize.height = std::min(minimalSize.height, image.size().height);
            }
        }

        return {transformations, minimalSize};
    }
}


export std::vector<std::filesystem::path> alignImages(const std::filesystem::path& dir, std::span<const std::filesystem::path> images)
{
    // TODO: replace with structure binding when supported by compilers
    const std::pair transformationsAndSize = calculateTransformations(images);
    const auto transformations = transformationsAndSize.first;
    const auto minimalSize = transformationsAndSize.second;

    const auto& first = images.front();
    const auto referenceImage = cv::imread(first.string());
    const cv::Rect firstImageSize(0, 0, minimalSize.width, minimalSize.height);
    const auto targetRect = calculateCrop(firstImageSize, transformations);
    const auto imagesCount = images.size();

    std::vector<std::filesystem::path> alignedImages;
    alignedImages.resize(imagesCount);

    #pragma omp parallel for
    for (size_t i = 0; i < imagesCount; i++)
    {
        const auto& imagePath = images[i];
        const auto imageFilename = imagePath.filename().string();

        const auto image = cv::imread(imagePath.string());

        // align
        cv::Mat imageAligned;
        if (i == 0)
            imageAligned = image;  // reference image does not need any transformations
        else
            cv::warpPerspective(image, imageAligned, transformations[i], image.size(), cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);

        // apply crop
        const auto croppedNextImg = imageAligned(targetRect);

        // save
        const auto path = dir / imageFilename;
        cv::imwrite(path.string(), croppedNextImg);

        alignedImages[i] = path;
    }

    return alignedImages;
}
