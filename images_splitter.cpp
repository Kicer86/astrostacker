
module;

#include <filesystem>
#include <span>
#include <vector>

#include <opencv2/opencv.hpp>


export module images_splitter;
import utils;


export std::vector<std::vector<std::filesystem::path>> splitImages(const std::filesystem::path& dir, std::span<const std::filesystem::path> images, std::pair<int, int> split)
{
    const int framesInSegmentToBeTaken = split.first;
    const int framesInSegmentToBeIgnored = split.second;
    const int segmentSize = framesInSegmentToBeTaken + framesInSegmentToBeIgnored;

    std::vector<std::vector<std::filesystem::path>> segments;

    // split images into groups of size 'segmentSize'
    for (auto it = images.begin(); it != images.end();)
    {
        auto next_it = std::distance(it, images.end()) >= segmentSize ? it + segmentSize : images.end();
        segments.emplace_back(it, next_it);
        it = next_it;
    }

    // in each group leave only first 'framesInSegmentToBeTaken' frames
    for (auto& segment: segments)
        if (segment.size() > framesInSegmentToBeTaken)
            segment.resize(framesInSegmentToBeTaken);

    return segments;
}
