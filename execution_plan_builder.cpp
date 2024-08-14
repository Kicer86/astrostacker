
module;

#include <filesystem>
#include <functional>
#include <span>
#include <vector>

export module execution_plan_builder;
import ifile_manager;
import utils;


export using ImagesList = std::vector<std::filesystem::path>;
export using ImagesView = std::span<const std::filesystem::path>;
export using Operation = std::function<ImagesList(const Utils::WorkingDir &, ImagesView)>;

export class ExecutionPlanBuilder
{
public:
    ExecutionPlanBuilder(const Utils::WorkingDir& wd, const IFileManager& fileManager, size_t maxSteps = std::numeric_limits<size_t>::max())
        : m_wd(wd)
        , m_fileManager(fileManager)
        , m_maxSteps(maxSteps == 0? std::numeric_limits<size_t>::max(): maxSteps)
    {

    }

    template<typename Op, typename... Args>
    requires std::is_invocable_r_v<ImagesList, Op, const std::filesystem::path &, ImagesView, Args...>
    auto addStep(std::string_view title, std::string_view dirName, Op op, Args... input)
    {
        using namespace std::placeholders;

        auto step = [this, op, input...](const Utils::WorkingDir& wd, ImagesView images)
        {
            return op(wd.path(), images, input...);
        };

        Operation operation = std::bind(step, _1, _2);
        m_ops.emplace_back(title, operation, dirName);
    }

    template<typename Op, typename... Args>
    requires std::is_invocable_r_v<ImagesList, Op, const std::filesystem::path &, ImagesView, Args...>
    auto addPostStep(std::string_view title, std::string_view dirName, Op op, Args... input)
    {
        using namespace std::placeholders;

        auto step = [this, op, input...](const Utils::WorkingDir& wd, ImagesView images)
        {
            return op(wd.path(), images, input...);
        };

        Operation operation = std::bind(step, _1, _2);
        m_postOps.emplace_back(title, operation, dirName);
    }

    ImagesList execute(ImagesView files)
    {
        ImagesList imagesList(files.begin(), files.end());
        size_t steps = m_maxSteps;

        std::optional<Utils::WorkingDir> previousWorkingDir;

        for(const auto& op: m_ops)
        {
            const auto& name = std::get<0>(op);
            const auto& func = std::get<1>(op);
            const auto& subdir = std::get<2>(op);
            const auto wd = m_wd.getSubDir(subdir);

            imagesList = Utils::measureTimeWithMessage(name, func, wd, imagesList);

            if (previousWorkingDir)
                m_fileManager.remove(*previousWorkingDir);

            previousWorkingDir = wd;

            steps--;
            if (steps == 0)
                break;
        }

        for(const auto& op: m_postOps)
        {
            const auto& name = std::get<0>(op);
            const auto& func = std::get<1>(op);
            const auto& subdir = std::get<2>(op);
            const auto wd = m_wd.getSubDir(subdir);

            imagesList = Utils::measureTimeWithMessage(name, func, wd, imagesList);
        }


        return imagesList;
    }

private:
    using Op = std::tuple<std::string, Operation, std::string>;
    std::vector<Op> m_ops;
    std::vector<Op> m_postOps;
    Utils::WorkingDir m_wd;
    const IFileManager& m_fileManager;
    const size_t m_maxSteps;
};
