
module;

#include <filesystem>
#include <vector>
#include <opencv2/opencv.hpp>

export module images_aligner;


namespace
{
    cv::Rect commonPart(cv::Rect2f commonPart, const std::vector<cv::Mat>& transformations)
    {
        for(const auto& transformation: transformations)
        {
            if (transformation.at<float>(0, 2) < 0)
                commonPart.x = transformation.at<float>(0, 2) * -1;

            if (transformation.at<float>(0, 2) > 0)
                commonPart.width -= transformation.at<float>(0, 2);

            if (transformation.at<float>(1, 2) < 0)
                commonPart.y = transformation.at<float>(1, 2) * -1;

            if (transformation.at<float>(1, 2) > 0)
                commonPart.height -= transformation.at<float>(1, 2);
        }

        return commonPart;
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

    std::vector<cv::Mat> calculateTransformations(const std::vector<std::filesystem::path>& images)
    {
        const cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);

        const auto& first = images.front();
        const auto referenceImage = cv::imread(first);

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

        const int imagesCount = static_cast<int>(images.size());
        transformations.resize(imagesCount);

        #pragma omp parallel for
        for (int i = 1; i < imagesCount; i++)
        {
            const auto& next = images[i];
            const auto image = cv::imread(next);

            cv::Mat imageGray;
            cv::cvtColor(image, imageGray, cv::COLOR_RGB2GRAY);

            const auto transformation = findTransformation(referenceImageGray, imageGray);

            transformations[i] = transformation;
        }

        return transformations;
    }
}


export void alignImages(const std::vector<std::filesystem::path>& images, const std::filesystem::path& dir)
{
    const auto transformations = calculateTransformations(images);

    const auto& first = images.front();
    const auto referenceImage = cv::imread(first);
    const cv::Rect firstImageSize(0, 0, referenceImage.size().width, referenceImage.size().height);
    const auto targetRect = commonPart(firstImageSize, transformations);

    #pragma omp parallel for
    for (int i = 0; i < images.size(); i++)
    {
        // read image
        const auto image = cv::imread(images[i]);

        // align
        cv::Mat imageAligned;
        if (i == 0)
            imageAligned = image;  // reference image does not need any transformations
        else
            cv::warpPerspective(image, imageAligned, transformations[i], image.size(), cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);

        // apply crop
        const auto croppedNextImg = imageAligned(targetRect);

        // save
        const std::string path = std::format("{}/{}.tiff", dir.native(), i);
        cv::imwrite(path, croppedNextImg);
    }
}
