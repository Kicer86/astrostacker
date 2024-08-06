
module;

#include <filesystem>
#include <iostream>
#include <optional>
#include <boost/program_options.hpp>

export module config;
import images_picker;


namespace
{
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
}


export struct Config
{
    const std::vector<std::filesystem::path> inputFiles;
    const std::filesystem::path wd;
    const std::optional<std::pair<int, int>> crop;
    const std::optional<std::pair<int, int>> split;
    const PickerMethod pickerMethod;
    const size_t skip;
    const size_t stopAfter;
    const int backgroundThreshold;
    const bool doObjectDetection;
    const bool collect;
};


export Config readParams(int argc, char** argv)
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
        ("use-best", po::value<std::string>()->default_value("median"), "Define how to choose best frames. Possible arguments: 'median', number (1÷100%)")
        ("stop-after", po::value<size_t>()->default_value(0), "Stop processing after N steps. For 0 (default) process all")
        ("transparent-background", po::value<int>()->default_value(-1), "Post step: replace black regions with transparent after all steps (see --stop-after) are finished. Provide threshold as argument (0-255)")
        ("collect", "Post step: copy results from last step into final directory")
        ("input-files", po::value<std::vector<std::string>>(), "input files");

    po::variables_map vm;
    po::positional_options_description p;
    p.add("input-files", -1);
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        std::stringstream help;
        help << desc;
        throw std::runtime_error(help.str());
    }

    if (vm.count("working-dir") == 0)
        throw std::invalid_argument("--working-dir option is required");

    if (vm.count("input-files") == 0)
        throw std::invalid_argument("Provide input files");

    const std::filesystem::path wd_option = vm["working-dir"].as<std::string>();
    const auto crop = readCrop(vm["crop"]);
    const auto split = readSegments(vm["split"]);
    const auto skip = vm["skip"].as<size_t>();
    const auto best = vm["use-best"];
    const bool doObjectDetection = vm.count("disable-object-detection") == 0;
    const std::vector<std::string> inputFilesStr = vm["input-files"].as<std::vector<std::string>>();
    const auto stopAfter = vm["stop-after"].as<size_t>();
    const auto backgroundThreshold = vm["transparent-background"].as<int>();
    const auto collect = vm.count("collect") > 0;

    const std::vector<std::filesystem::path> inputFiles(inputFilesStr.begin(), inputFilesStr.end());
    const auto pickerMethod = readPickerMethod(best);
    const auto wd = wd_option / getCurrentTime();

    if (pickerMethod.has_value() == false)
        throw std::invalid_argument("Invalid value for --use-best argument: " + best.as<std::string>() + ". Expected 'median' or % value 1÷100");

    return Config {
        .inputFiles = inputFiles,
        .wd = wd,
        .crop = crop,
        .split = split,
        .pickerMethod = *pickerMethod,
        .skip = skip,
        .stopAfter = stopAfter,
        .backgroundThreshold = backgroundThreshold,
        .doObjectDetection = doObjectDetection,
        .collect = collect,
    };
}
