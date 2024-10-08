
module;

#include <filesystem>
#include <format>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <spdlog/spdlog.h>

export module frame_extractor;
import utils;

namespace
{
    std::vector<std::filesystem::path> extractFrames(const std::filesystem::path& file, const std::filesystem::path& dir, size_t firstFrame, size_t lastFrame)
    {
        assert(lastFrame >= firstFrame);

        const auto count = lastFrame - firstFrame;
        if (count == 0)
            return {};

        const auto fileName = file.filename().string();

        std::vector<std::filesystem::path> paths;
        paths.reserve(static_cast<size_t>(count));

        cv::VideoCapture video(file.string(), cv::CAP_FFMPEG);
        if (video.isOpened())
        {
            video.set(cv::CAP_PROP_POS_FRAMES, static_cast<double>(firstFrame));

            for(size_t frame = firstFrame; frame < lastFrame; frame++)
            {
                cv::Mat frameMat;
                video >> frameMat;

                const std::filesystem::path path = dir / std::format("{}-{}.png", fileName, frame);
                cv::imwrite(path.string(), frameMat);
                paths.push_back(path);
            }
        }

        return paths;
    }
}


export size_t videoFrames(const std::filesystem::path& file)
{
    cv::VideoCapture video(file.string(), cv::CAP_FFMPEG);

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


export std::vector<std::filesystem::path> extractFrames(const std::filesystem::path& dir, std::span<const std::filesystem::path> files, size_t firstFrame, size_t lastFrame)
{
    const auto file = files.front();
    const auto frames = lastFrame - firstFrame;

    std::vector<std::pair<size_t, size_t>> segments;
    std::vector<std::filesystem::path> paths;

    // split frames among threads. It would be nice to ure regular 'parallel for' but each thread needs to get
    // continous region to work with.
    #pragma omp parallel
    {
        const auto threads = static_cast<size_t>(omp_get_num_threads());

        #pragma omp master
        {
            segments = Utils::split({firstFrame, lastFrame}, threads);
            paths.resize(frames);
        }
        #pragma omp barrier

        const auto thread = static_cast<size_t>(omp_get_thread_num());

        if (thread < segments.size())
        {
            const auto& [threadFirstFrame, threadLastFrame] = segments[thread];

            spdlog::debug("Thread #{} got frames {} - {} ({} frames)", thread, threadFirstFrame, threadLastFrame - 1, threadLastFrame - threadFirstFrame);

            const auto thread_paths = extractFrames(file, dir, threadFirstFrame, threadLastFrame);

            for(size_t out_f = threadFirstFrame, in_f = 0; out_f < threadLastFrame; out_f++, in_f++)
                paths[out_f - firstFrame] = thread_paths[in_f];
        }
        else
            spdlog::warn("Thread {} has nothing to do", thread);
    }

    return paths;
}
