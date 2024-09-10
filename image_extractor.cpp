
module;

#include <algorithm>
#include <filesystem>
#include <ranges>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <libraw/libraw.h>
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

    struct RawProcessorWrapper final
    {
        RawProcessorWrapper() {}
        ~RawProcessorWrapper()
        {
            rawProcessor.recycle();
        }

        LibRaw* operator->()
        {
            return &rawProcessor;
        }

        LibRaw& operator*()
        {
            return rawProcessor;
        }

    private:
        LibRaw rawProcessor;
    };


    // Function to apply noise reduction
    cv::Mat ReduceNoise(const cv::Mat& img)
    {
        cv::Mat denoisedImage;
        // Apply Non-Local Means denoising to reduce chromatic noise
        cv::fastNlMeansDenoisingColored(img, denoisedImage, 10, 10, 7, 21);  // You can tune the parameters
        return denoisedImage;
    }

    // Function to adjust contrast and brightness
    cv::Mat AdjustBrightnessContrast(const cv::Mat& img, double alpha, int beta)
    {
        cv::Mat newImage = cv::Mat::zeros(img.size(), img.type());
        // Apply contrast (alpha) and brightness (beta) adjustments
        img.convertTo(newImage, -1, alpha, beta);
        return newImage;
    }

    // Function to boost color saturation
    cv::Mat BoostSaturation(const cv::Mat& img, double saturationScale)
    {
        cv::Mat hsvImage;
        // Convert the image to HSV (Hue, Saturation, Value) color space
        cv::cvtColor(img, hsvImage, cv::COLOR_BGR2HSV);

        // Split the image into individual channels
        std::vector<cv::Mat> hsvChannels;
        cv::split(hsvImage, hsvChannels);

        // Scale the saturation channel (hsvChannels[1] is saturation)
        hsvChannels[1].convertTo(hsvChannels[1], -1, saturationScale, 0);

        // Merge channels back and convert to BGR color space
        cv::merge(hsvChannels, hsvImage);
        cv::Mat result;
        cv::cvtColor(hsvImage, result, cv::COLOR_HSV2BGR);

        return result;
    }


    // Function to perform CLAHE (Contrast Limited Adaptive Histogram Equalization)
    cv::Mat ApplyCLAHE(const cv::Mat& img)
    {
        cv::Mat labImage;
        // Convert the image to LAB color space
        cv::cvtColor(img, labImage, cv::COLOR_BGR2Lab);

        // Split the LAB image into channels
        std::vector<cv::Mat> labChannels;
        cv::split(labImage, labChannels);

        // Apply CLAHE to the L (lightness) channel
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8)); // CLAHE parameters
        clahe->apply(labChannels[0], labChannels[0]);

        // Merge the channels back and convert to BGR color space
        cv::merge(labChannels, labImage);
        cv::Mat result;
        cv::cvtColor(labImage, result, cv::COLOR_Lab2BGR);

        return result;
    }

    // Function to compute the mean saturation of an image
    double GetMeanSaturation(const cv::Mat& img)
    {
        cv::Mat hsvImage;
        cv::cvtColor(img, hsvImage, cv::COLOR_BGR2HSV);

        // Split into channels and calculate the mean of the Saturation channel
        std::vector<cv::Mat> hsvChannels;
        cv::split(hsvImage, hsvChannels);

        return cv::mean(hsvChannels[1])[0]; // Return mean saturation value
    }

    // Function to automatically adjust brightness, contrast, and saturation
    cv::Mat AutoAdjustImage(const cv::Mat& rawImg, const cv::Mat& referenceImg)
    {
        // Apply CLAHE for contrast enhancement
        cv::Mat adjustedImage = ApplyCLAHE(rawImg);

        // Calculate mean saturation of both images
        double rawMeanSaturation = GetMeanSaturation(adjustedImage);
        double referenceMeanSaturation = GetMeanSaturation(referenceImg);

        // Calculate the ratio to match the reference image's saturation
        double saturationScale = referenceMeanSaturation / rawMeanSaturation;

        // Boost the saturation to match the reference image
        adjustedImage = BoostSaturation(adjustedImage, saturationScale);

        return adjustedImage;
    }

    template<typename T, typename V>
    constexpr T saturate_cast(const V& v)
    {
        const auto clamped = std::clamp(v, static_cast<V>(std::numeric_limits<T>::min()), static_cast<V>(std::numeric_limits<T>::max()));

        return static_cast<T>(clamped);
    }


    std::ostream& operator<<(std::ostream& out, cv::Mat& m)
    {
        double min, max;
        cv::minMaxLoc(m, &min, &max);

        out << "size: " << m.cols << "x" << m.rows << "\n";
        out << "channels: " << m.channels() << ", bits per channel: " << m.elemSize1() * 8 << ", whole: " << m.elemSize() * 8 << "\n";
        out << "min: " << min << ", max: " << max << "\n";

        for (int i = 0; i < 20; i++)
            if (m.elemSize1() == 1)
                out << static_cast<int>(m.at<uint8_t>(0, i)) << " ";
            else if (m.elemSize1() == 2)
                out << m.at<uint16_t>(0, i) << " ";

        return out;
    }


    void normalizeRGBG(cv::Mat& rawImage)
    {
        assert(rawImage.cols % 2 == 0);
        assert(rawImage.rows % 2 == 0);

        for (int i = 0; i < rawImage.rows; i += 2)
            for (int j = 0; j < rawImage.cols; j += 2)
            {
                // Original "RGBG" pattern: (i,j) = R, (i,j+1) = G, (i+1,j) = G, (i+1,j+1) = B
                uint16_t R = rawImage.at<uint16_t>(i, j);           // Red
                uint16_t G1 = rawImage.at<uint16_t>(i, j + 1);      // Green (top)
                uint16_t B = rawImage.at<uint16_t>(i + 1, j);       // Blue
                uint16_t G2 = rawImage.at<uint16_t>(i + 1, j + 1);  // Green (bottom)

                // Rearrange to "RGGB" pattern
                rawImage.at<uint16_t>(i, j) = R;
                rawImage.at<uint16_t>(i, j + 1) = G1;
                rawImage.at<uint16_t>(i + 1, j) = G2;
                rawImage.at<uint16_t>(i + 1, j + 1) = B;
            }
    }


    cv::Mat performDemosaicing(const LibRaw& rawProcessor, cv::Mat& rawImage)
    {
        const char* bayer_pattern = &rawProcessor.imgdata.idata.cdesc[0];

        // Determine the appropriate OpenCV conversion code based on the Bayer pattern
        cv::ColorConversionCodes conversion_code;

        /*
        if (strncmp(bayer_pattern, "RGGB", 4) == 0)
            conversion_code = cv::COLOR_BayerRG2BGR;
        else if (strncmp(bayer_pattern, "BGGR", 4) == 0)
            conversion_code = cv::COLOR_BayerBG2BGR;
        else if (strncmp(bayer_pattern, "GRBG", 4) == 0)
            conversion_code = cv::COLOR_BayerGR2BGR;
        else if (strncmp(bayer_pattern, "GBRG", 4) == 0)
            conversion_code = cv::COLOR_BayerGB2BGR;
        else if (strncmp(bayer_pattern, "RGBG", 4) == 0)
        {
            normalizeRGBG(rawImage);
            conversion_code = cv::COLOR_BayerGB2BGR;
        }
        else
            throw std::runtime_error("Unknown Bayer pattern");
        */

        cv::Mat demosaiced_image;
        cv::cvtColor(rawImage, demosaiced_image, cv::COLOR_BayerBG2BGR);

        return demosaiced_image;
    }


    cv::Mat loadRawImage(const std::filesystem::path& path)
    {
        const auto pathStr = path.string();
        RawProcessorWrapper rawProcessor;

        if (rawProcessor->open_file(pathStr.c_str()) != LIBRAW_SUCCESS)
            throw std::runtime_error(std::format("Cannot open input file: {}", pathStr));

        // Adjust raw processor settings for better color handling
        rawProcessor->imgdata.params.use_camera_wb = 1; // Use camera white balance
        rawProcessor->imgdata.params.use_auto_wb = 1;   // Use auto white balance if camera WB is unavailable
        rawProcessor->imgdata.params.output_color = 1;  // Output color space: 1 = sRGB, 2 = Adobe RGB
        rawProcessor->imgdata.params.gamm[0] = 0.25f;   // Apply gamma correction (default for sRGB)
        rawProcessor->imgdata.params.gamm[1] = 4.5f;

        if (rawProcessor->unpack() != LIBRAW_SUCCESS)
            throw std::runtime_error(std::format("Error when unpacking RAW file data: {}", pathStr));


        // Get the raw image data
        //rawProcessor->raw2image();

        rawProcessor->dcraw_process();
        libraw_processed_image_t *image = rawProcessor->dcraw_make_mem_image();

        assert(image->bits == 8);
        assert(image->colors == 3);
        cv::Mat matImage(image->height, image->width, CV_8UC3, image->data);
        cv::Mat fixedImage;
        cv::cvtColor(matImage, fixedImage, cv::COLOR_RGB2BGR);

        LibRaw::dcraw_clear_mem(image);
        std::cout << fixedImage << std::endl;

        //fixedImage = ReduceNoise(fixedImage);

        fixedImage = AutoAdjustImage(fixedImage);

        return fixedImage;

         // ------------------- Step 3: White Balance Correction -------------------
        // Get white balance multipliers from metadata
        const float wb_red_multiplier = rawProcessor->imgdata.color.cam_mul[0];
        const float wb_green_multiplier = rawProcessor->imgdata.color.cam_mul[1];
        const float wb_blue_multiplier = rawProcessor->imgdata.color.cam_mul[2];

        // Apply white balance correction
        assert(fixedImage.elemSize1() == 1);
        for (int i = 0; i < fixedImage.rows; ++i)
            for (int j = 0; j < fixedImage.cols; ++j)
            {
                cv::Vec3b& pixel = fixedImage.at<cv::Vec3b>(i, j);
                pixel[0] = saturate_cast<uint8_t>(pixel[0] * wb_blue_multiplier);  // Blue channel
                pixel[1] = saturate_cast<uint8_t>(pixel[1] * wb_green_multiplier); // Green channel
                pixel[2] = saturate_cast<uint8_t>(pixel[2] * wb_red_multiplier);   // Red channel
            }

        std::cout << fixedImage << std::endl;

        return fixedImage;

        /*
        if (!image)
        {
            std::cerr << "Cannot process the image!" << std::endl;
            return -1;
        }

        // Convert the image to an OpenCV Mat
        cv::Mat matImage(image->height, image->width, CV_8UC3, image->data);

        // Get raw Bayer image dimensions and the data pointer
        const int width = rawProcessor->imgdata.sizes.iwidth;
        const int height = rawProcessor->imgdata.sizes.iheight;

        // Create OpenCV Mat to store the raw Bayer data (16-bit single-channel)
        cv::Mat raw_image(height, width, CV_16UC1, rawProcessor->imgdata.rawdata.raw_image);
        std::cout << raw_image << std::endl;

        // get own memory
        raw_image = raw_image.clone();
        std::cout << raw_image << std::endl;

        // ------------------- Step 1: Black Level Correction -------------------
        const int black_level = rawProcessor->imgdata.color.black;              // Black level value
        const int white_level = rawProcessor->imgdata.color.maximum;            // Maximum level (often 2^bit_depth - 1)

        double min, max;
        cv::minMaxLoc(raw_image, &min, &max);

        // scale values from range <black_level ÷ white_level> to <0 ÷ white_level>
        const double scale_factor = static_cast<double>(65535) / (white_level - black_level);
        for (int i = 0; i < raw_image.rows; ++i)
            for (int j = 0; j < raw_image.cols; ++j)
            {
                const int pixel_value = raw_image.at<uint16_t>(i, j);
                const int adjusted_value = std::max(pixel_value - black_level, 0);

                // Scale the value linearly from black_level - white_level to 0 - 65535
                raw_image.at<uint16_t>(i, j) = static_cast<uint16_t>(std::min(adjusted_value * scale_factor, 65535.0));
            }

        std::cout << raw_image << std::endl;

        // ------------------- Step 2: Demosaicing (Bayer to RGB conversion) -------------------
        // Use OpenCV's cvtColor to convert Bayer-patterned raw data to full RGB image
        cv::Mat demosaiced_image = performDemosaicing(*rawProcessor, raw_image);
        std::cout << demosaiced_image << std::endl;

        return demosaiced_image;

        // ------------------- Step 3: White Balance Correction -------------------
        // Get white balance multipliers from metadata
        const float wb_red_multiplier = rawProcessor->imgdata.color.cam_mul[0];
        const float wb_green_multiplier = rawProcessor->imgdata.color.cam_mul[1];
        const float wb_blue_multiplier = rawProcessor->imgdata.color.cam_mul[2];

        // Apply white balance correction
        assert(demosaiced_image.elemSize1() == 2);
        for (int i = 0; i < demosaiced_image.rows; ++i)
            for (int j = 0; j < demosaiced_image.cols; ++j)
            {
                cv::Vec3w& pixel = demosaiced_image.at<cv::Vec3w>(i, j);
                pixel[0] = saturate_cast<uint16_t>(pixel[0] * wb_blue_multiplier);  // Blue channel
                pixel[1] = saturate_cast<uint16_t>(pixel[1] * wb_green_multiplier); // Green channel
                pixel[2] = saturate_cast<uint16_t>(pixel[2] * wb_red_multiplier);   // Red channel
            }

        std::cout << demosaiced_image << std::endl;

        demosaiced_image.convertTo(demosaiced_image, CV_8UC3, 1.0 / 256.0);
        std::cout << demosaiced_image << std::endl;
        return demosaiced_image;

        // ------------------- Step 4: Gamma Correction -------------------
        // Convert image to floating point for gamma correction
        cv::Mat gamma_corrected_image;
        demosaiced_image.convertTo(gamma_corrected_image, CV_32F, 1.0 / 255.0);
        std::cout << gamma_corrected_image << std::endl;

        // Apply gamma correction (typical gamma is 2.2)
        cv::pow(gamma_corrected_image, 1.0 / 2.2, gamma_corrected_image);
        std::cout << gamma_corrected_image << std::endl;

        // Convert back to 16-bit
        gamma_corrected_image.convertTo(gamma_corrected_image, CV_16UC3, 255.0);
        std::cout << gamma_corrected_image << std::endl;

        // Convert to 8-bit
        gamma_corrected_image.convertTo(gamma_corrected_image, CV_8UC3, 1.0 / 256.0);
        std::cout << gamma_corrected_image << std::endl;

        return gamma_corrected_image;

        // ------------------- Step 5: Sharpening (Unsharp Mask) -------------------
        cv::Mat sharpened_image;
        cv::GaussianBlur(gamma_corrected_image, sharpened_image, cv::Size(0, 0), 3);
        cv::addWeighted(gamma_corrected_image, 1.5, sharpened_image, -0.5, 0, sharpened_image);

        std::cout << sharpened_image << std::endl;

        return sharpened_image;
        */
    }
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

    Utils::forEach(std::ranges::subrange(images.begin() + firstFrame, images.begin() + lastFrame), [&](const auto& i)
    {
        const auto& path = images[i];
        auto extension = path.extension().string();
        boost::algorithm::to_lower(extension);

        cv::Mat image;
        const auto pathStr = path.string();

        if (extension == ".nef")
        {
            image = loadRawImage(path);
        }
        else
            image = cv::imread(pathStr);

        const auto filename = path.stem();
        const auto newLocation = dir / (filename.string() + ".png");

        cv::imwrite(newLocation.string(), image);

        result[i] = newLocation;
    });

    return result;
}
