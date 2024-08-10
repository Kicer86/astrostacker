
module;

#include <filesystem>
#include <functional>
#include <span>
#include <vector>

export module execution_plan_builder;
import utils;


export using ImagesList = std::vector<std::filesystem::path>;
export using ImagesView = std::span<const std::filesystem::path>;
export using Operation = std::function<ImagesList(ImagesView)>;

export class ExecutionPlanBuilder
{
public:
    ExecutionPlanBuilder(const utils::WorkingDir& wd, size_t maxSteps = std::numeric_limits<size_t>::max())
        : m_wd(wd)
        , m_maxSteps(maxSteps == 0? std::numeric_limits<size_t>::max(): maxSteps)
    {

    }

    template<typename Op, typename... Args>
    requires std::is_invocable_r_v<ImagesList, Op, const std::filesystem::path &, ImagesView, Args...>
    auto addStep(std::string_view title, std::string_view dirName, Op op, Args... input)
    {
        using namespace std::placeholders;

        // purpose of this lambda is to postpone working dir creation until step execution
        auto step = [this, op, dirName, input...](ImagesView images)
        {
            return op(m_wd.getSubDir(dirName).path(), images, input...);
        };

        Operation operation = std::bind(step, _1);
        m_ops.emplace_back(title, operation);
    }

    template<typename Op, typename... Args>
    requires std::is_invocable_r_v<ImagesList, Op, const std::filesystem::path &, ImagesView, Args...>
    auto addPostStep(std::string_view title, std::string_view dirName, Op op, Args... input)
    {
        using namespace std::placeholders;

        // purpose of this lambda is to postpone working dir creation until step execution
        auto step = [this, op, dirName, input...](ImagesView images)
        {
            return op(m_wd.getSubDir(dirName).path(), images, input...);
        };

        Operation operation = std::bind(step, _1);
        m_postOps.emplace_back(title, operation);
    }

    ImagesList execute(ImagesView files)
    {
        ImagesList imagesList(files.begin(), files.end());
        size_t steps = m_maxSteps;

        for(const auto& op: m_ops)
        {
            imagesList = utils::measureTimeWithMessage(op.first, op.second, imagesList);

            steps--;
            if (steps == 0)
                break;
        }

        for(const auto& op: m_postOps)
            imagesList = utils::measureTimeWithMessage(op.first, op.second, imagesList);


        return imagesList;
    }

private:
    std::vector<std::pair<std::string, Operation>> m_ops;
    std::vector<std::pair<std::string, Operation>> m_postOps;
    utils::WorkingDir m_wd;
    const size_t m_maxSteps;
};
