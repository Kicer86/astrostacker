
#include <iostream>
#include <filesystem>

import frame_extractor;


int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3)
    {
        std::cerr << "Provide input video file and working dir as arguments\n";
        return 1;
    }

    const std::string_view input_file = argv[1];
    const std::string_view wd = argv[2];

    std::filesystem::create_directory(wd);

    const std::string extractedFramesDir = std::string(wd) + "/images";
    std::filesystem::create_directory(extractedFramesDir);

    extractFrames(input_file, extractedFramesDir);

    return 0;
}
