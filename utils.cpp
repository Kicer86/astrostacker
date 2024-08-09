
module;

#include <concepts>
#include <filesystem>
#include <format>
#include <ranges>
#include <span>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>


export module utils;

class Timer
{
    public:
        Timer(): start_time(std::chrono::high_resolution_clock::now()) {}

        double stop()
        {
            auto end_time = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(end_time - start_time).count();
        }

    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};

template<typename Func, typename... Args>
auto measureTime(Func func, Args&&... args)
{
    Timer timer;
    auto result = func(std::forward<Args>(args)...);
    double elapsed_time = timer.stop();
    return std::make_pair(elapsed_time, result);
}


export template <typename Func, typename... Args>
auto measureTimeWithMessage(std::string_view startMessage, Func func, Args&&... args)
{
    spdlog::info(startMessage);
    auto [time, result] = measureTime(func, std::forward<Args>(args)...);
    spdlog::info("Execution time: {}ms", time);
    return result;
}


export template<typename T, typename C>
void forEach(T items, C&& c)
{
    const int size = static_cast<int>(items.size());
    std::exception_ptr exception = nullptr;

    #pragma omp parallel for                        // TODO: visual studio requires int for loops, clean this up in the future
    for(int i = 0; i < size; i++)
    {
        try
        {
            c(static_cast<size_t>(i));
        }
        catch (...)
        {
            exception = std::current_exception();
#ifdef MSVC
            break;
#else
            #pragma omp cancel for
#endif
        }
    }

    if (exception)
        std::rethrow_exception(exception);
}


export template<typename T, std::size_t N>
requires std::invocable<T, const cv::Mat &> && (N > 0)
std::vector<std::filesystem::path> processImages(std::span<const std::filesystem::path> images, const std::array<std::filesystem::path, N>& dirs, T&& op)
{
    const auto imagesCount = images.size();
    std::vector<std::filesystem::path> resultPaths(imagesCount);

    forEach(images, [&](const size_t i)
    {
        const auto imagePath = images[i];
        const auto imageFilename = imagePath.filename().string();
        const cv::Mat image = cv::imread(imagePath.string());

        std::array<cv::Mat, N>  results;
        if constexpr (N == 1)
            results[0] = op(image);
        else
            results = op(image);

        std::optional<std::filesystem::path> firstPath;
        for (const auto [result, dir]: std::views::zip(results, dirs))
        {
            const auto path = dir / imageFilename;
            cv::imwrite(path.string(), result);

            if (!firstPath)
                firstPath = path;
        }

        resultPaths[i] = firstPath.value();
    });

    return resultPaths;
}


export template<typename T>
requires std::invocable<T, const cv::Mat &>
std::vector<std::filesystem::path> processImages(std::span<const std::filesystem::path> images, const std::filesystem::path& dir, T&& op)
{
    return processImages(images, std::array{dir}, op);
}


export void createLink(const std::filesystem::path& from, const std::filesystem::path& to)
{
    const auto targetPath = std::filesystem::relative(from, to.parent_path());
    try
    {
        std::filesystem::create_symlink(targetPath, to);
    }
    catch (const std::filesystem::filesystem_error &)
    {
        try
        {
            std::filesystem::copy_file(from, to);
        }
        catch (const std::filesystem::filesystem_error& err)
        {
            std::cerr << err.what() << "\n";
            throw;
        }
    }
}

export std::vector<std::filesystem::path> createLinks(std::span<const std::filesystem::path> from, const std::filesystem::path& to)
{
    const auto imagesCount = from.size();
    std::vector<std::filesystem::path> result;
    result.reserve(imagesCount);

    for (const auto& input: from)
    {
        const auto inputName = input.filename().string();
        const auto newPath = to / inputName;
        createLink(input, newPath);

        result.push_back(newPath);
    }

    return result;
}

export class WorkingDir
{
public:
    explicit WorkingDir(std::filesystem::path dir)
        : m_dir(dir)
    {}

    WorkingDir(const WorkingDir &) = default;
    WorkingDir(WorkingDir &&) = default;

    WorkingDir& operator=(const WorkingDir &) = default;
    WorkingDir& operator=(WorkingDir &&) = default;

    WorkingDir getSubDir(std::string_view subdir)
    {
        const std::filesystem::path path = m_dir / std::format("#{} {}", m_c, subdir);
        std::filesystem::create_directories(path);
        m_c++;

        return WorkingDir(path);
    }

    WorkingDir getExactSubDir(std::string_view subdir) const
    {
        const std::filesystem::path path = m_dir / subdir;
        std::filesystem::create_directories(path);

        return WorkingDir(path);
    }

    std::filesystem::path path() const
    {
        return m_dir;
    }

private:
    std::filesystem::path m_dir;
    int m_c = 0;
};


export size_t divideWithRoundUp(size_t lhs, size_t rhs)
{
    return (lhs + rhs - 1) / rhs;
}

namespace utils
{
export std::vector<std::pair<size_t, size_t>> split(const std::pair<size_t, size_t>& input, std::size_t groups)
{
    const auto& first = input.first;
    const auto& last = input.second;
    const auto elements = last - first;
    const auto groupSize = divideWithRoundUp(elements, groups);
    std::vector<std::pair<size_t, size_t>> result;

    for (size_t i = 0; i < groups; i++)
    {
        const auto groupFirst = std::min(last, first + groupSize * i);
        const auto groupLast = std::min(last, groupFirst + groupSize);

        if ((groupLast - groupFirst) > 0)
            result.emplace_back(groupFirst, groupLast);
    }

    return result;
}
}
