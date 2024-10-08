
module;

#include <filesystem>
#include <vector>

#include <opencv2/opencv.hpp>

export module aberration_fixer;
import utils;

namespace
{
    cv::Mat enhanceContrast(const cv::Mat& image)
    {
        cv::Mat enhanced;
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->apply(image, enhanced);
        return enhanced;
    }

    cv::Mat alignChannel(const cv::Mat& referenceChannel, const cv::Mat& channel)
    {
        cv::Mat enhancedRef = enhanceContrast(referenceChannel);
        cv::Mat enhancedCh = enhanceContrast(channel);

        // ORB detector and matcher
        cv::Ptr<cv::ORB> orb = cv::ORB::create();
        std::vector<cv::KeyPoint> kpRef, kp;
        cv::Mat desRef, des;

        // Detect and compute features for reference and target channels
        orb->detectAndCompute(enhancedRef, cv::noArray(), kpRef, desRef);
        orb->detectAndCompute(enhancedCh, cv::noArray(), kp, des);

        // Brute-force matcher with Hamming distance
        cv::BFMatcher bf(cv::NORM_HAMMING, true);
        std::vector<cv::DMatch> matches;
        bf.match(desRef, des, matches);
        std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; });

        // Extract matched points
        std::vector<cv::Point2f> srcPts, dstPts;
        for (size_t i = 0; i < matches.size(); ++i)
        {
            srcPts.push_back(kp[matches[i].trainIdx].pt);
            dstPts.push_back(kpRef[matches[i].queryIdx].pt);
        }

        // Find homography and warp the channel
        const cv::Mat homography = cv::findHomography(srcPts, dstPts, cv::RANSAC);

        cv::Mat alignedChannel;
        cv::warpPerspective(channel, alignedChannel, homography, referenceChannel.size());

        return alignedChannel;
    }
}


export std::vector<std::filesystem::path> fixChromaticAberration(const std::filesystem::path& dir, std::span<const std::filesystem::path> images, bool debug)
{
    const auto rDir = dir / "_red";
    const auto gDir = dir / "_green";
    const auto bDir = dir / "_blue";
    const auto fDir = dir / "fixed";

    const std::array dirs{fDir, rDir, gDir, bDir};
    const auto fixed = Utils::processImages(images, dirs, debug, [](const auto& image)
    {
        // Split the image into B, G, R channels
        std::vector<cv::Mat> channels(3);
        cv::split(image, channels);

        const auto& b = channels[0];
        const auto& g = channels[1];
        const auto& r = channels[2];

        const cv::Mat alignedR = alignChannel(g, r);
        const cv::Mat alignedB = alignChannel(g, b);

        // Merge the aligned channels back into one image
        std::vector<cv::Mat> alignedChannels = { alignedB, g, alignedR };

        cv::Mat correctedImage;
        cv::merge(alignedChannels, correctedImage);

        return std::array{correctedImage, r, g, b};
    });

    return fixed;
}
