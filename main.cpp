
#include <filesystem>
#include <iostream>

import abberation_fixer;
import config;
import execution_plan_builder;
import frame_extractor;
import images_aligner;
import images_cropper;
import images_enhancer;
import images_picker;
import images_splitter;
import images_stacker;
import object_localizer;
import utils;


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
                const auto imageSegments = splitImages(images, *split);
                const auto segments = imageSegments.size();
                for (size_t i = 0; i < segments; i++)
                {
                    auto segmentDir = wd.getExactSubDir(std::to_string(i));
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
