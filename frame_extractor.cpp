
module;

#include <chrono>
#include <string_view>
#include <opencv2/opencv.hpp>

export module frame_extractor;


export void extractVideo(std::string_view file, std::string_view dir)
{
    cv::VideoCapture video(std::string(file), cv::CAP_FFMPEG);

    if (video.isOpened())
    {
        video.set(cv::CAP_PROP_POS_FRAMES, 0);
        const auto frames = video.get(cv::CAP_PROP_FRAME_COUNT);

        for(double frame = 0; frame < frames; frame++)
        {
            const auto start = std::chrono::high_resolution_clock::now();

            cv::Mat frameMat;
            video >> frameMat;

            cv::imwrite(std::format("{}/{}.tiff", dir, std::to_string(static_cast<int64_t>(frame))), frameMat);
            const auto stop = std::chrono::high_resolution_clock::now();
            const auto duration = stop - start;
            const auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);

            std::cout << "Processed frame #" << frame << ". Time: " << duration_ms.count() << "ms" << std::endl;
        }
    }
}
