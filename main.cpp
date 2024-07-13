
#include <iostream>
#include <filesystem>

import frame_extractor;
import images_aligner;
import images_enhancer;
import images_picker;
import images_stacker;
import object_localizer;
import utils;


namespace
{
    std::filesystem::path makeSubDir(const std::filesystem::path& wd, std::string_view subdir)
    {
        static int c = 0;
        const std::filesystem::path path = wd / std::format("#{} {}", c, subdir);
        std::filesystem::create_directory(path);
        c++;

        return path;
    }

    std::string getCurrentTime()
    {
        const std::time_t now = std::time(nullptr);
        const std::tm *ptm = std::localtime(&now);

        std::ostringstream oss;
        oss << std::put_time(ptm, "%Y%m%d-%H%M%S");

        return oss.str();
    }
}


int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Provide input video file and working dir as arguments\n";
        return 1;
    }

    const std::filesystem::path input_file = argv[1];
    const std::filesystem::path wd = std::filesystem::path(argv[2]) / getCurrentTime();

    std::filesystem::create_directory(wd);

    const auto extractedFramesDir = makeSubDir(wd, "images");
    const auto images = measureTimeWithMessage("Extracting frames from video.", extractFrames, input_file, extractedFramesDir);

    const auto objectDir = makeSubDir(wd, "object");
    const auto objects =  measureTimeWithMessage("Extracting main object.", extractObject, images, objectDir);

    const auto bestImagesDir = makeSubDir(wd, "best");
    const auto bestImages = measureTimeWithMessage("Choosing best images.", pickImages, objects, bestImagesDir);

    const auto alignedImagesDir = makeSubDir(wd, "aligned");
    const auto alignedImages = measureTimeWithMessage("Aligning images.", alignImages, bestImages, alignedImagesDir);

    const auto stackedImagesDir = makeSubDir(wd, "stacked");
    const auto stackedImages = measureTimeWithMessage("Stacking images.", stackImages, alignedImages, stackedImagesDir);

    const auto enahncedImagesDir = makeSubDir(wd, "enhanced");
    measureTimeWithMessage("Enhancing images.", enhanceImages, stackedImages, enahncedImagesDir);

    return 0;
}
