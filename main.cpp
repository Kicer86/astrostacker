
#include <iostream>
#include <filesystem>

#include <boost/program_options.hpp>


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
    namespace po = boost::program_options;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("working-dir", po::value<std::string>(), "set working directory")
        ("input-files", po::value<std::vector<std::string>>(), "input files");

    po::variables_map vm;
    po::positional_options_description p;
    p.add("input-files", -1);
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        return 1;
    }

    if (vm.count("working-dir") == 0)
    {
        std::cerr << "--working-dir option is required\n";
        return 1;
    }

    if (vm.count("input-files") == 0)
    {
        std::cerr << "Provide input files\n";
        return 1;
    }

    const std::filesystem::path wd_option = vm["working-dir"].as<std::string>();
    const std::vector<std::string> inputFiles = vm["input-files"].as<std::vector<std::string>>();
    const std::filesystem::path input_file = inputFiles[0];
    const std::filesystem::path wd = wd_option / getCurrentTime();

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
