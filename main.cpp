
#include <iostream>
#include <filesystem>

import images_aligner;
import images_picker;
import frame_extractor;


int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3)
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
    const auto bestImages = pickImages(images);

    std::cout << "\nPicked images (" << bestImages.size() << "):\n";
    for(const auto& bestImage: bestImages)
        std::cout << bestImage << "\n";

    const std::filesystem::path alignedImagesDir = wd / "aligned";
    alignImages(bestImages, alignedImagesDir);

    return 0;
}
