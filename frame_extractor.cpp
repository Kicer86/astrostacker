
module;

#include <chrono>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <omp.h>

export module frame_extractor;

namespace
{
    int divideWithRoundUp(int lhs, int rhs)
    {
        return (lhs + rhs - 1) / rhs;
    }

    int videoFrames(const std::filesystem::path& file)
    {
        cv::VideoCapture video(file, cv::CAP_FFMPEG);

        if (video.isOpened())
        {
            const auto frames = video.get(cv::CAP_PROP_FRAME_COUNT);
            return static_cast<int>(frames);
        }
        else
            return 0;
    }

    std::vector<std::string> extractFrames(const std::filesystem::path& file, const std::filesystem::path& dir, int firstFrame, int lastFrame)
    {
        const auto count = lastFrame - firstFrame;
        if (count == 0)
            return {};

        assert(count > 0 );

        std::vector<std::string> paths;
        paths.reserve(count);

        cv::VideoCapture video(std::string(file), cv::CAP_FFMPEG);
        if (video.isOpened())
        {
            video.set(cv::CAP_PROP_POS_FRAMES, firstFrame);

            for(int frame = firstFrame; frame < lastFrame; frame++)
            {
                cv::Mat frameMat;
                video >> frameMat;

                const std::string path = std::format("{}/{}.tiff", dir.native(), frame);
                cv::imwrite(path, frameMat);
                paths.push_back(path);
            }
        }

        return paths;
    }
}

export std::vector<std::filesystem::path> extractFrames(const std::filesystem::path& file, const std::filesystem::path& dir)
{
    const auto total_start = std::chrono::high_resolution_clock::now();
    const auto frames = videoFrames(file);

    if (frames == 0)
        return {};

    std::vector<std::filesystem::path> paths;

    #pragma omp parallel
    {
        const auto threads = omp_get_num_threads();
        const auto thread = omp_get_thread_num();
        const auto group_size = divideWithRoundUp(frames, threads);

        // Thread * frames-to-process-by-each-thread needs to be at least equal to number of frames
        assert(threads * group_size >= frames);

        #pragma omp master
        {
            paths.resize(frames);
            std::cout << "Starting extraction of " << frames << " frames with " << threads << " threads. Group size: " << group_size << "\n";
        }
        #pragma omp barrier

        const auto firstFrame = std::min(frames, group_size * thread);
        const auto lastFrame = std::min(frames, firstFrame + group_size);
        const auto thread_paths = extractFrames(file, dir, firstFrame, lastFrame);

        for(int out_f = firstFrame, in_f = 0; out_f < lastFrame; out_f++, in_f++)
            paths[out_f] = thread_paths[in_f];
    }

    const auto total_stop = std::chrono::high_resolution_clock::now();
    const auto duration = total_stop - total_start;
    const auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);

    std::cout << "Total time: " << duration_ms.count() << "ms" << std::endl;

    return paths;
}
