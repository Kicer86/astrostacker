
module;

#include <chrono>
#include <string_view>
#include <opencv2/opencv.hpp>
#include <omp.h>

export module frame_extractor;

namespace
{
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
}

export void extractVideo(std::string_view file, std::string_view dir)
{
    const auto total_start = std::chrono::high_resolution_clock::now();
    const auto frames = videoFrames(file);

    #pragma omp parallel
    {
        const auto threads = omp_get_num_threads();
        const auto thread = omp_get_thread_num();
        const auto group_size = frames > threads? frames / threads : 1;

        #pragma omp master
        {
            std::cout << "Starting extraction with " << threads << " threads. Group size: " << group_size << "\n";
        }
        #pragma omp barrier

        cv::VideoCapture video(std::string(file), cv::CAP_FFMPEG);
        if (video.isOpened())
        {
            const auto frames = static_cast<int64_t>(video.get(cv::CAP_PROP_FRAME_COUNT));
            const auto first_frame = group_size * thread;
            const auto last_frame = std::min(frames, first_frame + group_size);

            video.set(cv::CAP_PROP_POS_FRAMES, first_frame);

            for(int64_t frame = first_frame; frame < last_frame; frame++)
            {
                const auto start = std::chrono::high_resolution_clock::now();

                cv::Mat frameMat;
                video >> frameMat;

                cv::imwrite(std::format("{}/{}.tiff", dir, std::to_string(static_cast<int64_t>(frame))), frameMat);
                const auto stop = std::chrono::high_resolution_clock::now();
                const auto duration = stop - start;
                const auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);

                std::stringstream thread_message;
                thread_message << "Processed frame #" << frame << ". Time: " << duration_ms.count() << "ms\n";
                std::cout << thread_message.str();
            }
        }
    }

    const auto total_stop = std::chrono::high_resolution_clock::now();
    const auto duration = total_stop - total_start;
    const auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);

    std::cout << "Total time: " << duration_ms.count() << "ms" << std::endl;
}
