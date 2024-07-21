
module;

#include <filesystem>
#include <format>
#include <opencv2/opencv.hpp>
#include <omp.h>

export module frame_extractor;

namespace
{
    size_t divideWithRoundUp(size_t lhs, size_t rhs)
    {
        return (lhs + rhs - 1) / rhs;
    }

    size_t videoFrames(const std::filesystem::path& file)
    {
        cv::VideoCapture video(file.string(), cv::CAP_ANY);

        if (video.isOpened())
        {
            const auto frames = video.get(cv::CAP_PROP_FRAME_COUNT);
            
            if (frames >= 0)
                return static_cast<size_t>(frames);
            else
                return 0;
        }
        else
            return 0;
    }

    std::vector<std::string> extractFrames(const std::filesystem::path& file, const std::filesystem::path& dir, size_t firstFrame, size_t lastFrame)
    {
        assert(lastFrame >= firstFrame);

        const auto count = lastFrame - firstFrame;
        if (count == 0)
            return {};

        std::vector<std::string> paths;
        paths.reserve(static_cast<size_t>(count));

        cv::VideoCapture video(file.string(), cv::CAP_ANY);
        if (video.isOpened())
        {
            video.set(cv::CAP_PROP_POS_FRAMES, static_cast<double>(firstFrame));

            for(size_t frame = firstFrame; frame < lastFrame; frame++)
            {
                cv::Mat frameMat;
                video >> frameMat;

                const std::string path = std::format("{}/{}.png", dir.string(), frame);
                cv::imwrite(path, frameMat);
                paths.push_back(path);
            }
        }

        return paths;
    }
}

export std::vector<std::filesystem::path> extractFrames(const std::filesystem::path& file, const std::filesystem::path& dir)
{
    const auto frames = videoFrames(file);

    if (frames == 0)
        return {};

    std::vector<std::filesystem::path> paths;

    #pragma omp parallel
    {
        const auto threads = static_cast<size_t>(omp_get_num_threads());
        const auto thread = static_cast<size_t>(omp_get_thread_num());
        const auto group_size = divideWithRoundUp(frames, threads);

        // Thread * frames-to-process-by-each-thread needs to be at least equal to number of frames
        assert(threads * group_size >= frames);

        #pragma omp master
        {
            paths.resize(static_cast<size_t>(frames));
        }
        #pragma omp barrier

        const auto firstFrame = std::min(frames, group_size * thread);
        const auto lastFrame = std::min(frames, firstFrame + group_size);
        const auto thread_paths = extractFrames(file, dir, firstFrame, lastFrame);

        for(size_t out_f = firstFrame, in_f = 0; out_f < lastFrame; out_f++, in_f++)
            paths[out_f] = thread_paths[in_f];
    }

    return paths;
}
