
module;

#include <filesystem>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

export module images_enhancer;
import utils;


namespace
{
    cv::Mat richardsonLucyDeconvolution(const cv::Mat& image, const cv::Mat& psf, int iterations) {
        cv::Mat estimate = image.clone();
        cv::Mat estimatePrevious;
        cv::Mat psfFlipped;
        cv::flip(psf, psfFlipped, -1);

        for (int i = 0; i < iterations; i++)
        {
            cv::filter2D(estimate, estimatePrevious, -1, psfFlipped, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
            cv::divide(image, estimatePrevious, estimatePrevious);
            cv::filter2D(estimatePrevious, estimatePrevious, -1, psf, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
            cv::multiply(estimate, estimatePrevious, estimate);
        }

        return estimate;
    }

    cv::Mat enhanceContrast(const cv::Mat& img)
    {
        cv::Mat labImage;
        cv::cvtColor(img, labImage, cv::COLOR_BGR2Lab);

        std::vector<cv::Mat> labPlanes(3);
        cv::split(labImage, labPlanes);

        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(4.0);
        cv::Mat claheImage;
        clahe->apply(labPlanes[0], claheImage);

        claheImage.copyTo(labPlanes[0]);
        cv::merge(labPlanes, labImage);

        cv::Mat result;
        cv::cvtColor(labImage, result, cv::COLOR_Lab2BGR);
        return result;
    }

    cv::Mat reduceNoise(const cv::Mat& img)
    {
        cv::Mat denoised;
        cv::fastNlMeansDenoisingColored(img, denoised, 10, 10, 7, 21);
        return denoised;
    }

    cv::Mat sharpenImage(const cv::Mat& img)
    {
        cv::Mat blurred, sharpened;
        cv::GaussianBlur(img, blurred, cv::Size(0, 0), 3);
        cv::addWeighted(img, 1.5, blurred, -0.5, 0, sharpened);
        return sharpened;
    }
}

export std::vector<std::filesystem::path> enhanceImages(const std::filesystem::path& dir, std::span<const std::filesystem::path> images)
{
    const auto result = Utils::processImages(images, dir, [](const cv::Mat& image)
    {
        cv::Mat psf = cv::getGaussianKernel(21, 5, CV_32F);
        psf = psf * psf.t();

        const cv::Mat deconvolvedImage = richardsonLucyDeconvolution(image, psf, 10);
        const cv::Mat contrastEnhancedImage = enhanceContrast(deconvolvedImage);
        const cv::Mat denoisedImage = reduceNoise(contrastEnhancedImage);
        const cv::Mat sharpenedImage = sharpenImage(contrastEnhancedImage);

        return sharpenedImage;
    });

    return result;
}
