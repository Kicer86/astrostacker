
module;

#include <algorithm>
#include <filesystem>
#include <ranges>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/process.hpp>
#include <opencv2/opencv.hpp>


export module image_extractor;
import utils;

namespace
{
    auto collectImages(const std::filesystem::path& inputDir)
    {
        auto files =
            std::filesystem::recursive_directory_iterator(inputDir) |
            std::views::filter([](const std::filesystem::directory_entry &entry)
            {
                return entry.is_regular_file();
            }) |
            std::views::transform([](const std::filesystem::directory_entry &entry)
            {
                return entry.path();
            });

        return files | std::ranges::to<std::vector>();
    }

    class ProcessRunner
    {
        boost::filesystem::path m_processPath;
        std::string_view m_process;

    public:
        ProcessRunner(std::string_view process)
            : m_process(process)
        {
            m_processPath = boost::process::search_path(m_process);
        }

        template<typename... Args>
        int execute(Args&& ...args)
        {
            if (m_processPath.empty())
                throw std::runtime_error(std::format("Could not locate {} executable", m_process));
            else
                return boost::process::system(m_processPath, std::forward<Args>(args)...);
        }
    };
}


export size_t countImages(const std::filesystem::path& inputDir)
{
    return collectImages(inputDir).size();
}

export std::vector<std::filesystem::path> collectImages(const std::filesystem::path& dir, std::span<const std::filesystem::path> files, size_t firstFrame, size_t lastFrame)
{
    if (files.size() != 1)
        throw std::runtime_error("Unexpected number of input elements: " + std::to_string(files.size()));

    const auto& input = files.front();

    if (not std::filesystem::is_directory(input))
        throw std::runtime_error("Input path: " + input.string() + " is not a directory");

    const auto images = collectImages(input);
    const auto count = images.size();

    if (count < lastFrame)
        throw std::out_of_range("last frame > number of frames");

    std::vector<std::filesystem::path> result;
    result.resize(lastFrame - firstFrame);

    ProcessRunner runner("darktable-cli");

    Utils::forEach(std::ranges::subrange(images.begin() + firstFrame, images.begin() + lastFrame), [&](const auto& i)
    {
        const auto& path = images[i];
        auto extension = path.extension().string();
        boost::algorithm::to_lower(extension);

        const auto pathStr = path.string();
        const auto filename = path.stem();
        const auto newLocation = dir / (filename.string() + ".png");
        const auto newLocationStr = newLocation.string();

        if (extension == ".nef")
        {
            const auto wd = boost::filesystem::temp_directory_path() / "astro-stacker" / boost::filesystem::unique_path();
            boost::filesystem::create_directories(wd);

            runner.execute(pathStr, newLocationStr, "--core", "--configdir", wd.string());
            boost::filesystem::remove_all(wd);
        }
        else
        {
            const cv::Mat image = cv::imread(pathStr);
            cv::imwrite(newLocationStr, image);
        }

        result[i] = newLocation;
    });

    return result;
}
