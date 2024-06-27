
#include <iostream>
#include <filesystem>

import frame_extractor;
import images_aligner;
import images_enhancer;
import images_picker;
import images_stacker;
import object_localizer;


namespace
{
    std::filesystem::path makeSubDir(const std::filesystem::path& wd, std::string_view subdir)
    {
        const std::filesystem::path path = wd / subdir;
        std::filesystem::create_directory(path);

        return path;
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
    const std::filesystem::path wd = argv[2];

    std::filesystem::create_directory(wd);

    const auto extractedFramesDir = makeSubDir(wd, "images");
    const auto images = extractFrames(input_file, extractedFramesDir);

    const auto objectDir = makeSubDir(wd, "object");
    const auto objects = extractObject(images, objectDir);

    const auto bestImages = pickImages(objects);

    const auto alignedImagesDir = makeSubDir(wd, "aligned");
    const auto alignedImages = alignImages(bestImages, alignedImagesDir);

    const auto stackedImagesDir = makeSubDir(wd, "stacked");
    const auto stackedImages = stackImages(alignedImages, stackedImagesDir);

    const auto enahncedImagesDir = makeSubDir(wd, "enhanced");
    enhanceImages(stackedImages, enahncedImagesDir);

    return 0;
}
