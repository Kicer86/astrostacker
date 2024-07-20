
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

    std::optional<std::pair<int, int>> readCrop(const boost::program_options::variable_value& cropValue)
    {
        if (cropValue.empty())
            return {};
        else
        {
            const auto input = cropValue.as<std::string>();
            const size_t pos = input.find('x');
            if (pos == std::string::npos)
                return {};

            const int width = std::stoi(input.substr(0, pos));
            const int height = std::stoi(input.substr(pos + 1));

            return std::pair{width, height};
        }
    }

    template<typename First, typename... Rest>
    const First& getFirst(const First& first, Rest... rest)
    {
        return first;
    }

    template<typename... Args>
    auto step(std::string_view title, const std::filesystem::path& wd, std::string_view subdir, auto op, Args... input)
    {
        const auto stepWorkingDir = makeSubDir(wd, subdir);
        return measureTimeWithMessage(title, op, std::forward<Args>(input)..., stepWorkingDir);
    }

    template<typename... Args>
    auto stepIf(bool condition, std::string_view title, const std::filesystem::path& wd, std::string_view subdir, auto op, Args... inputs)
    {
        if (condition)
            return step(title, wd, subdir, op, std::forward<Args>(inputs)...);
        else
            return getFirst(std::forward<Args>(inputs)...);
    }
}


int main(int argc, char** argv)
{
    namespace po = boost::program_options;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("working-dir", po::value<std::string>(), "set working directory")
        ("crop", po::value<std::string>(), "crop images to given size. Example: 1000x800")
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
    const auto crop = readCrop(vm["crop"]);
    const std::vector<std::string> inputFiles = vm["input-files"].as<std::vector<std::string>>();
    const std::filesystem::path input_file = inputFiles[0];
    const std::filesystem::path wd = wd_option / getCurrentTime();

    if (std::filesystem::exists(input_file) == false)
    {
        std::cerr << "Could not open file: " << input_file << "\n";
        return 1;
    }

    const bool wd_status = std::filesystem::create_directories(wd);

    if (wd_status == false)
    {
        std::cerr << "Could not create working dir: " << wd << "\n";
        return 1;
    }

    const auto images = step("Extracting frames from video.", wd, "images", extractFrames, input_file);
    if (images.empty())
    {
        std::cerr << "Error reading frames from video file.\n";
        return 1;
    }

    const auto objects =  step("Extracting main object.", wd, "object", extractObject, images);
    const auto cropped = stepIf(crop.has_value(), "Cropping.", wd, "crop", cropImages, objects, crop.has_value() ? (*crop) : std::pair<int, int>());
    const auto bestImages = step("Choosing best images.", wd, "best", pickImages, cropped);
    const auto alignedImages = step("Aligning images.", wd, "aligned", alignImages, bestImages);
    const auto stackedImages = step("Stacking images.", wd, "stacked", stackImages, alignedImages);
    step("Enhancing images.", wd, "enhanced", enhanceImages, stackedImages);

    return 0;
}
