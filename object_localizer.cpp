
module;

#include <filesystem>
#include <vector>
#include <opencv2/opencv.hpp>

export module object_localizer;

import utils;


namespace
{
    std::pair<cv::Mat, cv::Mat> findBrightestObject(const cv::Mat& img)
    {
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        // Apply a threshold to find bright regions
        cv::Mat binary;
        double maxVal;
        cv::minMaxLoc(gray, nullptr, &maxVal);
        cv::threshold(gray, binary, maxVal * 0.1, 255, cv::THRESH_BINARY);

        cv::Mat contoursImg = img.clone();

        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Find the largest contour
        double maxArea = 0;
        std::optional<std::size_t> maxAreaIdx;
        for (size_t i = 0; i < contours.size(); i++)
        {
            cv::rectangle(contoursImg, cv::boundingRect(contours[i]), {0, 255, 0}, 1);

            const double area = cv::contourArea(contours[i]);
            if (area > maxArea)
            {
                maxArea = area;
                maxAreaIdx = i;
            }
        }

        if (maxAreaIdx.has_value() == false)
        {
            std::cerr << "Error: No contours found." << std::endl;
            return {};
        }

        // Compute bounding box with margin
        cv::Rect bbox = cv::boundingRect(contours[*maxAreaIdx]);
        const int hmargin = static_cast<int>(bbox.width * 0.05);
        const int vmargin = static_cast<int>(bbox.width * 0.05);
        bbox.x = std::max(bbox.x - hmargin, 0);
        bbox.y = std::max(bbox.y - vmargin, 0);
        bbox.width = std::min(bbox.width + 2 * hmargin, img.cols - bbox.x);
        bbox.height = std::min(bbox.height + 2 * vmargin, img.rows - bbox.y);

        const cv::Mat object = img(bbox);
        return {object, contoursImg};
    }
}


export std::vector<std::filesystem::path> extractObject(const std::vector<std::filesystem::path>& images, const std::filesystem::path& dir)
{
    const auto contoursDir = dir / "contours";
    const auto objectsDir = dir / "objects";

    std::filesystem::create_directory(contoursDir);
    std::filesystem::create_directory(objectsDir);

    const auto extractedObjects = processImages(images, std::array{objectsDir, contoursDir}, [](const cv::Mat& image)
    {
        const auto [object, contours] = findBrightestObject(image);

        return std::array{object, contours};
    });

    return extractedObjects;
}
