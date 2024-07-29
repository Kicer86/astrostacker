
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
    requires std::is_invocable_r_v<ImagesList, Op, const std::filesystem::path &, ImagesView, Args...>
    auto addStep(std::string_view title, std::string_view dirName, Op op, Args... input)
    {
        using namespace std::placeholders;
        Operation operation = std::bind(op, m_wd.getSubDir(dirName).path(), _1, std::forward<Args>(input)...);
        m_ops.emplace_back(title, operation);
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
