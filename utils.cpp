
module;

#include <concepts>
#include <filesystem>
#include <format>
#include <ranges>
#include <span>
#include <opencv2/opencv.hpp>

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
    std::cout << startMessage << std::flush;
    auto [time, result] = measureTime(func, std::forward<Args>(args)...);
    std::cout << " Execution time: " << time << " ms" << std::endl;
    return result;
}


export template<typename T, typename C>
void forEach(T items, C&& c)
{
    const int size = static_cast<int>(items.size());

    #pragma omp parallel for                        // TODO: visual studio requires int for loops, clean this up in the future
    for(int i = 0; i < size; i++)
        c(static_cast<size_t>(i));
}


export template<typename T, std::size_t N>
requires std::invocable<T, const cv::Mat &> && (N > 0)
std::vector<std::filesystem::path> processImages(const std::vector<std::filesystem::path>& images, const std::array<std::filesystem::path, N>& dirs, T&& op)
{
    const auto imagesCount = images.size();
    std::vector<std::filesystem::path> resultPaths(imagesCount);

    forEach(images, [&](const size_t i)
    {
        const cv::Mat image = cv::imread(images[i].string());

        std::array<cv::Mat, N>  results;
        if constexpr (N == 1)
            results[0] = op(image);
        else
            results = op(image);

        std::optional<std::filesystem::path> firstPath;
        for (const auto [result, dir]: std::views::zip(results, dirs))
        {
            const auto path = dir / std::format("{}.tiff", i);
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
std::vector<std::filesystem::path> processImages(const std::vector<std::filesystem::path>& images, const std::filesystem::path& dir, T&& op)
{
    return processImages(images, std::array{dir}, op);
}
