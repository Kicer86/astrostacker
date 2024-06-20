
module;

#include <chrono>
#include <string_view>
#include <opencv2/opencv.hpp>
#include <omp.h>

export module frame_extractor;

namespace
{
    int64_t divideWithRoundUp(int64_t lhs, int64_t rhs)
    {
        return (lhs + rhs - 1) / rhs;
    }

    int64_t videoFrames(std::string_view file)
    {
        cv::VideoCapture video(std::string(file), cv::CAP_FFMPEG);

        if (video.isOpened())
        {
            video.set(cv::CAP_PROP_POS_FRAMES, 0);
            const auto frames = video.get(cv::CAP_PROP_FRAME_COUNT);
            return frames;
        }
        else
            return 0;
    }

    void extractFrames(std::string_view file, std::string_view dir, int64_t firstFrame, int64_t lastFrame)
    {
        cv::VideoCapture video(std::string(file), cv::CAP_FFMPEG);
        if (video.isOpened())
        {
            video.set(cv::CAP_PROP_POS_FRAMES, firstFrame);

            for(int64_t frame = firstFrame; frame < lastFrame; frame++)
            {
                cv::Mat frameMat;
                video >> frameMat;

                cv::imwrite(std::format("{}/{}.tiff", dir, std::to_string(static_cast<int64_t>(frame))), frameMat);
            }
        }
    }
}

export void extractVideo(std::string_view file, std::string_view dir)
{
    const auto total_start = std::chrono::high_resolution_clock::now();
    const auto frames = videoFrames(file);

    if (frames == 0)
        return;

    #pragma omp parallel
    {
        const auto threads = omp_get_num_threads();
        const auto thread = omp_get_thread_num();
        const auto group_size = divideWithRoundUp(frames, threads);

        // Thread * frames-to-process-by-each-thread needs to be at least equal to number of frames
        assert(threads * group_size >= frames);

        #pragma omp master
        {
            std::cout << "Starting extraction of " << frames << " frames with " << threads << " threads. Group size: " << group_size << "\n";
        }
        #pragma omp barrier

        const auto firstFrame = group_size * thread;
        const auto lastFrame = std::min(frames, firstFrame + group_size);

        extractFrames(file, dir, firstFrame, lastFrame);
    }

    const auto total_stop = std::chrono::high_resolution_clock::now();
    const auto duration = total_stop - total_start;
    const auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);

    std::cout << "Total time: " << duration_ms.count() << "ms" << std::endl;
}
