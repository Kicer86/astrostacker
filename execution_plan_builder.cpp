
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
    ExecutionPlanBuilder(const WorkingDir& wd)
        : m_wd(wd)
    {

    }

    template<typename Op, typename... Args>
    requires std::is_invocable_r_v<ImagesList, Op, const std::filesystem::path &, ImagesView, Args...> ||
                std::is_invocable_r_v<ImagesList, Op, WorkingDir, ImagesView, Args...>
    auto addStep(std::string_view title, std::string_view dirName, Op op, Args... input)
    {
        using namespace std::placeholders;

        if constexpr (std::is_invocable_r_v<ImagesList, Op, const std::filesystem::path &, ImagesView, Args...>)
        {
            Operation operation = std::bind(op, m_wd.getSubDir(dirName).path(), _1, std::forward<Args>(input)...);
            m_ops.emplace_back(title, operation);
        }
        else
        {
            Operation operation = std::bind(op, m_wd.getSubDir(dirName), _1, std::forward<Args>(input)...);
            m_ops.emplace_back(title, operation);
        }
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
    requires std::is_invocable_r_v<ImagesList, Op, ImagesView, Args...> ||
                std::is_invocable_r_v<ImagesList, Op, WorkingDir, ImagesView, Args...>
    auto addStep(Op op, Args... input)
    {
        using namespace std::placeholders;

        if constexpr (std::is_invocable_r_v<ImagesList, Op, ImagesView, Args...>)
        {
            Operation operation = std::bind(op, _1, std::forward<Args>(input)...);
            m_ops.emplace_back(std::optional<std::string>(), operation);
        }
        else
        {
            Operation operation = std::bind(op, m_wd, _1, std::forward<Args>(input)...);
            m_ops.emplace_back(std::optional<std::string>(), operation);
        }
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
