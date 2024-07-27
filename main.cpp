
#include <cassert>
#include <filesystem>
#include <iostream>

import config;
import frame_extractor;
import images_aligner;
import images_cropper;
import images_enhancer;
import images_picker;
import images_splitter;
import images_stacker;
import object_localizer;
import utils;


namespace
{
    template<typename First, typename... Rest>
    const First& getFirst(const First& first, Rest... rest)
    {
        return first;
    }

    template<typename... Args>
    auto step(std::string_view title, const WorkingDir& wd, auto op, Args... input)
    {
        return measureTimeWithMessage(title, op, wd.path(), std::forward<Args>(input)...);
    }

    template<typename... Args>
    auto step(std::string_view title, auto op, Args... input)
    {
        return measureTimeWithMessage(title, op, std::forward<Args>(input)...);
    }
}


int main(int argc, char** argv)
{
    try
    {
        const auto config = readParams(argc, argv);

        WorkingDir wd(config.wd);
        const auto& input_file = config.inputFiles.front();
        const auto& skip = config.skip;
        const auto& split = config.split;
        const auto& doObjectDetection = config.doObjectDetection;
        const auto& crop = config.crop;
        const auto& pickerMethod = config.pickerMethod;

        const auto images = step("Extracting frames from video.", wd.getSubDir("images"), extractFrames, input_file);
        if (images.empty())
            throw std::runtime_error("Error reading frames from video file.");

        const std::vector<std::filesystem::path> imagesAfterSkip(images.begin() + skip, images.end());

        const auto imageSegments = split.has_value()? step("Splitting.", splitImages, imagesAfterSkip, *split) : std::vector<std::vector<std::filesystem::path>>{imagesAfterSkip};
        const auto segments = imageSegments.size();
        if (segments < 1)
            throw std::logic_error("Calculated number of segments is less than 1: " + std::to_string(segments));

        // if there is more than one segment, create subdirs structure
        auto segmentsDir = segments == 1? wd : wd.getSubDir("segments");

        for (size_t i = 0; i < segments; i++)
        {
            WorkingDir segment_wd = segments == 1? segmentsDir : segmentsDir.getExactSubDir(std::to_string(i));

            const auto& segmentImages = imageSegments[i];
            if (segments > 1)
                std::cout << "Processing segment #" << i + 1 << " of " << segments << "\n";

            const auto objects = doObjectDetection? step("Extracting main object.", segment_wd.getSubDir("object"), extractObject, segmentImages) : segmentImages;
            const auto cropped = crop.has_value()? step("Cropping.", segment_wd.getSubDir("crop"), cropImages, objects, *crop) : objects;
            const auto bestImages = step("Choosing best images.", segment_wd.getSubDir("best"), pickImages, cropped, pickerMethod);
            const auto alignedImages = step("Aligning images.", segment_wd.getSubDir("aligned"), alignImages, bestImages);
            const auto stackedImages = step("Stacking images.", segment_wd.getSubDir("stacked"), stackImages, alignedImages);
            step("Enhancing images.", segment_wd.getSubDir("enhanced"), enhanceImages, stackedImages);
        }
    }
    catch(const std::runtime_error& error)
    {
        std::cout << error.what() << "\n";
        return 1;
    }
    catch(const std::invalid_argument& error)
    {
        std::cout << "Error: " << error.what() << "\n";
        return 1;
    }
    catch(const std::logic_error& error)
    {
        std::cout << "Error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
