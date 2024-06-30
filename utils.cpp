
module;

#include <concepts>
#include <filesystem>
#include <opencv2/opencv.hpp>

export module utils;

namespace
{
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

    template <typename Func, typename... Args>
    auto measureTime(Func func, Args&&... args)
    {
        Timer timer;
        auto result = func(std::forward<Args>(args)...);
        double elapsed_time = timer.stop();
        return std::make_pair(elapsed_time, result);
    }
}


export template<typename T>
requires std::invocable<T, const cv::Mat &>
std::vector<std::filesystem::path> processImages(const std::vector<std::filesystem::path>& images, const std::filesystem::path& dir, T&& op)
{
    const auto imagesCount = images.size();
    std::vector<std::filesystem::path> resultPaths(imagesCount);

    #pragma omp parallel for
    for(size_t i = 0; i < imagesCount; i++)
    {
        const cv::Mat image = cv::imread(images[i]);
        const auto result = op(image);

        const auto path = dir / std::format("{}.tiff", i);
        cv::imwrite(path, result);

        resultPaths[i] = path;
    }

    return resultPaths;
}


export template <typename Func, typename... Args>
auto measureTimeWithMessage(std::string_view startMessage, Func func, Args&&... args)
{
    std::cout << startMessage << std::flush;
    auto [time, result] = measureTime(func, std::forward<Args>(args)...);
    std::cout << " Execution time: " << time << " ms" << std::endl;
    return result;
}
