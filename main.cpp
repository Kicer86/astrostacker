
#include <iostream>
#include <filesystem>

import frame_extractor;
import images_aligner;
import images_picker;
import images_stacker;
import object_localizer;


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

    const std::filesystem::path extractedFramesDir = wd / "images";
    std::filesystem::create_directory(extractedFramesDir);

    const auto images = extractFrames(input_file, extractedFramesDir);

    const std::filesystem::path objectDir = wd / "object";
    std::filesystem::create_directory(objectDir);
    const auto objects = extractObject(images, objectDir);

    const auto bestImages = pickImages(objects);

    const std::filesystem::path alignedImagesDir = wd / "aligned";
    std::filesystem::create_directory(alignedImagesDir);

    const auto alignedImages = alignImages(bestImages, alignedImagesDir);

    const std::filesystem::path stackedImagesDir = wd / "stacked";
    std::filesystem::create_directory(stackedImagesDir);

    stackImages(alignedImages, stackedImagesDir);

    return 0;
}
