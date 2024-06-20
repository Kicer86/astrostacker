
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

    extractFrames(input_file, wd);

    return 0;
}
