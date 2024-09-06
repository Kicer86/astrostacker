
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/cfg/env.h>
#include <omp.h>
#include <indicators/dynamic_progress.hpp>
#include <indicators/indeterminate_progress_bar.hpp>


import aberration_fixer;
import config;
import execution_plan_builder;
import file_manager;
import frame_extractor;
import image_extractor;
import images_aligner;
import images_cropper;
import images_enhancer;
import images_picker;
import images_splitter;
import images_stacker;
import object_localizer;
import transparency_applier;
import utils;


namespace
{
    size_t countInputImages(const std::filesystem::path& input)
    {
        if (std::filesystem::is_directory(input))
            return countImages(input);
        else
            return videoFrames(input);
    }

    std::vector<std::filesystem::path> extractImages(const std::filesystem::path& dir, std::span<const std::filesystem::path> files, size_t firstFrame, size_t lastFrame)
    {
        if (files.size() != 1)
            throw std::runtime_error("Unexpected number of input elements: " + std::to_string(files.size()));

        const auto& input = files.front();

        if (std::filesystem::is_directory(input))
            return collectImages(dir, files, firstFrame, lastFrame);
        else
            return extractFrames(dir, files, firstFrame, lastFrame);
    }

    class HideCursor final
    {
    public:
        HideCursor()
        {
            indicators::show_console_cursor(false);
        }

        ~HideCursor()
        {
            indicators::show_console_cursor(true);
        }
    };
}


int main(int argc, char** argv)
{
    spdlog::cfg::load_env_levels();
    HideCursor _;

    try
    {
        const auto config = Config::readParams(argc, argv);

        Utils::WorkingDir wd(config.wd);
        const auto& inputFiles = config.inputFiles;
        const auto& skip = config.skip;
        const auto& split = config.split;
        const auto& doObjectDetection = config.doObjectDetection;
        const auto& crop = config.crop;
        const auto& pickerMethod = config.pickerMethod;
        const auto& stopAfter = config.stopAfter;
        const auto& backgroundThreshold = config.backgroundThreshold;
        const auto& threads = config.threads;
        const auto& debugSteps = config.debugSteps;
        const auto& cleanup = config.cleanup;

        const auto maxThreads = omp_get_max_threads();
        auto useThreads = threads > 0? threads: maxThreads + threads;
        useThreads = std::clamp(useThreads, 1, maxThreads);

        spdlog::info("Using {} threads", useThreads);
        omp_set_num_threads(useThreads);

        const auto& inputFile = config.inputFiles.front();

        const size_t firstFrame = skip;
        const size_t lastFrame = countInputImages(inputFile);
        const size_t frames = lastFrame - firstFrame;

        if (frames == 0)
            throw std::runtime_error("Error reading frames from video file.");

        const size_t framesInSegmentToBeTaken = split? split->first : frames;
        const size_t framesInSegmentToBeIgnored = split? split->second : 0;
        const size_t segmentSize = framesInSegmentToBeTaken + framesInSegmentToBeIgnored;
        const size_t segments = Utils::divideWithRoundUp(frames, segmentSize);

        const FileManager fm(cleanup);

        indicators::show_console_cursor(false);
        indicators::DynamicProgress<indicators::IndeterminateProgressBar> progressBarManager;
        progressBarManager.set_option(indicators::option::HideBarWhenComplete{false});
        std::vector<std::pair<int, std::filesystem::path>> allImages;

        for(int i = 0; i < segments; i++)
        {
            spdlog::info("Processing segment {} of {}", i + 1, segments);
            const auto segmentBegin = i * segmentSize;
            const auto segmentEnd = std::min(segmentBegin + segmentSize, lastFrame);

            Utils::WorkingDir segmentWorkingDir = segments == 1? wd : wd.getExactSubDir(std::to_string(i + 1));

            ExecutionPlanBuilder epb(segmentWorkingDir, progressBarManager, fm, stopAfter);
            epb.addStep("Acquiring input images.", "images", extractImages, segmentBegin, segmentEnd);

            if (doObjectDetection)
                epb.addStep("Extracting main object.", "object", extractObject, debugSteps);

            if (crop.has_value())
                epb.addStep("Cropping.", "crop", cropImages, *crop);

            epb.addStep("Fixing chromatic abberation", "chroma", fixChromaticAberration, debugSteps);
            epb.addStep("Choosing best images.", "best", pickImages, pickerMethod);
            epb.addStep("Aligning images.", "aligned", alignImages);
            epb.addStep("Stacking images.", "stacked", stackImages);
            epb.addStep("Enhancing images.", "enhanced", enhanceImages);

            if (backgroundThreshold >= 0)
                epb.addPostStep("Applying transparency.", "transparent", applyTransparency, backgroundThreshold);

            const auto segmentFiles = epb.execute(inputFiles);
            for (const auto& path: segmentFiles)
                allImages.emplace_back(i, path);
        }

        if (config.collect && segments > 1)
        {
            const auto allPath = wd.path() / "all";
            std::filesystem::create_directory(allPath);

            for (const auto& srcInfo: allImages)
            {
                const auto& srcPath = srcInfo.second;
                const auto srcFileName = srcPath.filename();
                const auto outputPath = allPath / (std::to_string(srcInfo.first) + "_" + srcFileName.string());
                Utils::copyFile(srcPath, outputPath);
            }
        }
    }
    catch (const std::runtime_error& error)
    {
        std::cout << error.what() << "\n";
        return 1;
    }
    catch (const std::invalid_argument& error)
    {
        spdlog::error(error.what());
        return 1;
    }
    catch (const std::logic_error& error)
    {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
    catch (const cv::Exception& error)
    {
        spdlog::error(error.what());
        return 1;
    }
    catch(...)
    {
        spdlog::error("Fail: Unhandled exception");
        return 1;
    }

    return 0;
}
