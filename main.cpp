
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/cfg/env.h>
#include <omp.h>


import aberration_fixer;
import config;
import execution_plan_builder;
import file_manager;
import frame_extractor;
import images_aligner;
import images_cropper;
import images_enhancer;
import images_picker;
import images_splitter;
import images_stacker;
import object_localizer;
import transparency_applier;
import utils;


int main(int argc, char** argv)
{
    spdlog::cfg::load_env_levels();

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
        const size_t lastFrame = videoFrames(inputFile);
        const size_t frames = lastFrame - firstFrame;

        if (frames == 0)
            throw std::runtime_error("Error reading frames from video file.");

        const size_t framesInSegmentToBeTaken = split? split->first : frames;
        const size_t framesInSegmentToBeIgnored = split? split->second : 0;
        const size_t segmentSize = framesInSegmentToBeTaken + framesInSegmentToBeIgnored;
        const size_t segments = Utils::divideWithRoundUp(frames, segmentSize);

        const FileManager fm(cleanup);

        std::vector<std::pair<int, std::filesystem::path>> allImages;
        for(int i = 0; i < segments; i++)
        {
            spdlog::info("Processing segment {} of {}", i + 1, segments);
            const auto segmentBegin = i * segmentSize;
            const auto segmentEnd = std::min(segmentBegin + segmentSize, lastFrame);

            Utils::WorkingDir segmentWorkingDir = segments == 1? wd : wd.getExactSubDir(std::to_string(i));

            ExecutionPlanBuilder epb(segmentWorkingDir, fm, stopAfter);
            epb.addStep("Extracting frames from video.", "images", extractFrames, segmentBegin, segmentEnd);

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
