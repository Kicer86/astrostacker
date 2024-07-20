
#include <iostream>
#include <filesystem>

#include <boost/program_options.hpp>


import frame_extractor;
import images_aligner;
import images_cropper;
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

    auto step(std::string_view title, auto op, const auto& input, const std::filesystem::path& wd, std::string_view subdir)
    {
        const auto stepWorkingDir = makeSubDir(wd, subdir);
        return measureTimeWithMessage(title, op, input, stepWorkingDir);
    }

    auto step_if(bool condition, std::string_view title, auto op, const auto& input, const std::filesystem::path& wd, std::string_view subdir)
    {
        if (condition)
            return step(title, op, input, wd, subdir);
        else
            return input;
    }
}


int main(int argc, char** argv)
{
    namespace po = boost::program_options;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("working-dir", po::value<std::string>(), "set working directory")
        ("crop", "enable crop")
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
    const bool crop = vm.count("crop");
    const std::vector<std::string> inputFiles = vm["input-files"].as<std::vector<std::string>>();
    const std::filesystem::path input_file = inputFiles[0];
    const std::filesystem::path wd = wd_option / getCurrentTime();

    std::filesystem::create_directory(wd);

    const auto images = step("Extracting frames from video.", extractFrames, input_file, wd, "images");
    const auto objects =  step("Extracting main object.", extractObject, images, wd, "object");
    const auto cropped = step_if(crop, "Cropping.", cropImages, objects, wd, "crop");
    const auto bestImages = step("Choosing best images.", pickImages, cropped, wd, "best");
    const auto alignedImages = step("Aligning images.", alignImages, bestImages, wd, "aligned");
    const auto stackedImages = step("Stacking images.", stackImages, alignedImages, wd, "stacked");
    step("Enhancing images.", enhanceImages, stackedImages, wd, "enhanced");

    return 0;
}
