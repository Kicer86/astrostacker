
#include <filesystem>
#include <iostream>

#include <boost/program_options.hpp>


import frame_extractor;
import images_aligner;
import images_cropper;
import images_enhancer;
import images_picker;
import images_splitter;
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

    std::optional<std::pair<int, int>> readSegments(const boost::program_options::variable_value& cropValue)
    {
        if (cropValue.empty())
            return {};
        else
        {
            const auto input = cropValue.as<std::string>();
            const size_t pos = input.find(',');
            if (pos == std::string::npos)
                return {};

            const int length = std::stoi(input.substr(0, pos));
            const int gap = std::stoi(input.substr(pos + 1));

            return std::pair{length, gap};
        }
    }

    std::optional<PickerMethod> readPickerMethod(const boost::program_options::variable_value& pickerMethod)
    {
        const auto pickedMethod = pickerMethod.as<std::string>();

        if (pickedMethod == "median")
            return MedianPicker{};
        else if (const auto value = std::stoi(pickedMethod); value != 0)
            return value;
        else
            return {};
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
}


int main(int argc, char** argv)
{
    namespace po = boost::program_options;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("working-dir", po::value<std::string>(), "set working directory")
        ("crop", po::value<std::string>(), "crop images to given size. Example: --crop 1000x800")
        ("split", po::value<std::string>(), "Split video into segments. Provide segment lenght and gap in frames as argument. Example: --split 120,40")
        ("skip", po::value<size_t>()->default_value(0), "Skip n frames from the video begining. Example: --skip 60")
        ("disable-object-detection", "Disable object detection step")
        ("use-best", po::value<std::string>()->default_value("median"), "Define how to choose best frames. Possible arguments: median, number (1-100)")
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
    const auto split = readSegments(vm["split"]);
    const auto skip = vm["skip"].as<size_t>();
    const auto best = vm["use-best"];
    const bool doObjectDetection = vm.count("disable-object-detection") == 0;
    const std::vector<std::string> inputFiles = vm["input-files"].as<std::vector<std::string>>();
    const std::filesystem::path input_file = inputFiles[0];
    const std::filesystem::path wd = wd_option / getCurrentTime();
    const auto pickerMethod = readPickerMethod(best);

    if (pickerMethod.has_value() == false)
    {
        std::cerr << "Invalid value for --use-best argument: " << best.as<std::string>() << ". Expected 'median' or % value 1÷100\n";
        return 1;
    }

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

    const std::vector<std::filesystem::path> imagesAfterSkip(images.begin() + skip, images.end());

    const auto imageSegments = split.has_value()? step("Splitting.", wd, "segments", splitImages, imagesAfterSkip, *split) : std::vector<std::vector<std::filesystem::path>>{imagesAfterSkip};
    const auto segments = imageSegments.size();

    for (size_t i = 0; i < segments; i++)
    {
        const std::filesystem::path segment_wd = segments == 1? wd : wd / std::to_string(i);
        std::filesystem::create_directories(segment_wd);

        const auto& segmentImages = imageSegments[i];
        if (segments > 1)
            std::cout << "Processing segment #" << i + 1 << " of " << segments << "\n";

        const auto objects = doObjectDetection? step("Extracting main object.", segment_wd, "object", extractObject, segmentImages) : segmentImages;
        const auto cropped = crop.has_value()? step("Cropping.", segment_wd, "crop", cropImages, objects, *crop) : objects;
        const auto bestImages = step("Choosing best images.", segment_wd, "best", pickImages, cropped, *pickerMethod);
        const auto alignedImages = step("Aligning images.", segment_wd, "aligned", alignImages, bestImages);
        const auto stackedImages = step("Stacking images.", segment_wd, "stacked", stackImages, alignedImages);
        step("Enhancing images.", segment_wd, "enhanced", enhanceImages, stackedImages);
    }

    return 0;
}
