
#include <cassert>
#include <filesystem>
#include <functional>
#include <iostream>
#include <span>

import abberation_fixer;
import config;
import frame_extractor;
import images_aligner;
import images_cropper;
import images_enhancer;
import images_picker;
import images_splitter;
import images_stacker;
import object_localizer;
import utils;


namespace
{
    using ImagesList = std::vector<std::filesystem::path>;
    using ImagesView = std::span<const std::filesystem::path>;
    using Operation = std::function<ImagesList(ImagesView)>;

    class ExecutionPlanBuilder
    {
    public:
        ExecutionPlanBuilder(const WorkingDir& wd)
            : m_wd(wd)
        {

        }

        template<typename Op, typename... Args>
        requires std::is_invocable_r_v<ImagesList, Op, const std::filesystem::path &, ImagesView, Args...>
        auto addStep(std::string_view title, std::string_view dirName, Op op, Args... input)
        {
            using namespace std::placeholders;

            Operation operation = std::bind(op, m_wd.getSubDir(dirName).path(), _1, std::forward<Args>(input)...);
            m_ops.emplace_back(title, operation);
        }

        template<typename Op, typename... Args>
        requires std::is_invocable_r_v<ImagesList, Op, WorkingDir, ImagesView, Args...>
        auto addStep(std::string_view title, std::string_view dirName, Op op, Args... input)
        {
            using namespace std::placeholders;

            Operation operation = std::bind(op, m_wd.getSubDir(dirName), _1, std::forward<Args>(input)...);
            m_ops.emplace_back(title, operation);
        }

        template<typename Op, typename... Args>
        requires std::is_invocable_r_v<ImagesList, Op, ImagesView, Args...>
        auto addStep(std::string_view title, Op op, Args... input)
        {
            using namespace std::placeholders;

            Operation operation = std::bind(op, _1, std::forward<Args>(input)...);
            m_ops.emplace_back(title, operation);
        }

        template<typename Op, typename... Args>
        requires std::is_invocable_r_v<ImagesList, Op, ImagesView, Args...>
        auto addStep(Op op, Args... input)
        {
            using namespace std::placeholders;

            Operation operation = std::bind(op, _1, std::forward<Args>(input)...);
            m_ops.emplace_back(std::optional<std::string>(), operation);
        }

        template<typename Op, typename... Args>
        requires std::is_invocable_r_v<ImagesList, Op, WorkingDir, ImagesView, Args...>
        auto addStep(Op op, Args... input)
        {
            using namespace std::placeholders;

            Operation operation = std::bind(op, m_wd, _1, std::forward<Args>(input)...);
            m_ops.emplace_back(std::optional<std::string>(), operation);
        }

        ImagesList execute(ImagesView files)
        {
            ImagesList imagesList(files.begin(), files.end());

            for(const auto& op: m_ops)
                if (op.first)
                    imagesList = measureTimeWithMessage(*op.first, op.second, imagesList);
                else
                    imagesList = op.second(imagesList);

            return imagesList;
        }

    private:
        std::vector<std::pair<std::optional<std::string>, Operation>> m_ops;
        WorkingDir m_wd;
    };

    template<typename First, typename... Rest>
    const First& getFirst(const First& first, Rest... rest)
    {
        return first;
    }

    template<typename... Args>
    auto step(std::string_view title, const WorkingDir& wd, auto op, Args... input)
    {
        return measureTimeWithMessage(title, op, wd.path(), std::forward<Args>(input)...);
    }

    template<typename... Args>
    auto step(std::string_view title, auto op, Args... input)
    {
        return measureTimeWithMessage(title, op, std::forward<Args>(input)...);
    }
}


int main(int argc, char** argv)
{
    try
    {
        const auto config = readParams(argc, argv);

        WorkingDir wd(config.wd);
        const auto& inputFiles = config.inputFiles;
        const auto& skip = config.skip;
        const auto& split = config.split;
        const auto& doObjectDetection = config.doObjectDetection;
        const auto& crop = config.crop;
        const auto& pickerMethod = config.pickerMethod;

        ExecutionPlanBuilder epb(wd);
        epb.addStep("Extracting frames from video.", "images", extractFrames);

        if (skip > 0)
            epb.addStep("Skipping frames.", [skip](ImagesView images) -> ImagesList
            {
                return std::vector<std::filesystem::path>(images.begin() + skip, images.end());
            });

        auto onSegmentOperations = [&](WorkingDir wd, ImagesView images)
        {
            ExecutionPlanBuilder segmentEpb(wd);

            if (doObjectDetection)
                segmentEpb.addStep("Extracting main object.", "object", extractObject);

            if (crop.has_value())
                segmentEpb.addStep("Cropping.", "crop", cropImages, *crop);

            //segmentEpb.addStep("Fixing chromatic abberation", "chroma", fixChromaticAbberation);
            segmentEpb.addStep("Choosing best images.", "best", pickImages, pickerMethod);
            segmentEpb.addStep("Aligning images.", "aligned", alignImages);
            segmentEpb.addStep("Stacking images.", "stacked", stackImages);
            segmentEpb.addStep("Enhancing images.", "enhanced", enhanceImages);

            return segmentEpb.execute(images);
        };

        if (split.has_value())
            epb.addStep("Splitting.", "segments", [&onSegmentOperations, &split](WorkingDir wd, ImagesView images) -> ImagesList
            {
                auto segmentsDir = wd.getSubDir("segments");

                const auto imageSegments = splitImages(images, *split);
                const auto segments = imageSegments.size();
                for (size_t i = 0; i < segments; i++)
                {
                    auto segmentDir = segmentsDir.getExactSubDir(std::to_string(i));
                    onSegmentOperations(segmentDir, imageSegments[i]);
                }

                return ImagesList();
            });
        else
            epb.addStep(onSegmentOperations);

        epb.execute(inputFiles);
    }
    catch(const std::runtime_error& error)
    {
        std::cout << error.what() << "\n";
        return 1;
    }
    catch(const std::invalid_argument& error)
    {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
    catch(const std::logic_error& error)
    {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
    catch(...)
    {
        std::cerr << "Fail: Unhandled exception\n";
        return 1;
    }

    return 0;
}
