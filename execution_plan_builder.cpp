
module;

#include <filesystem>
#include <functional>
#include <span>
#include <vector>

#include <indicators/dynamic_progress.hpp>
#include <indicators/indeterminate_progress_bar.hpp>

export module execution_plan_builder;
import ifile_manager;
import utils;


export using ImagesList = std::vector<std::filesystem::path>;
export using ImagesView = std::span<const std::filesystem::path>;
export using Operation = std::function<ImagesList(const Utils::WorkingDir &, ImagesView)>;

export class ExecutionPlanBuilder
{
public:
    ExecutionPlanBuilder(const Utils::WorkingDir& wd, indicators::DynamicProgress<indicators::IndeterminateProgressBar>& progressBarManager, const IFileManager& fileManager, size_t maxSteps = std::numeric_limits<size_t>::max())
        : m_wd(wd)
        , m_progressBarManager(progressBarManager)
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

        auto execute = [&](const Op& op)
        {
            const auto& name = std::get<0>(op);
            const auto& func = std::get<1>(op);
            const auto& subdir = std::get<2>(op);
            const auto wd = m_wd.getSubDir(subdir);

            auto bar = std::make_unique<indicators::IndeterminateProgressBar>(
                //indicators::option::BarWidth{40},
                indicators::option::Start{"["},
                indicators::option::Fill{"·"},
                indicators::option::Lead{"<==>"},
                indicators::option::End{"]"},
                indicators::option::PostfixText{name}
                //indicators::option::ForegroundColor{indicators::Color::yellow},
                //indicators::option::FontStyles{
                //    std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}
            );

            const auto barIdx = m_progressBarManager.push_back(std::move(bar));

            std::jthread progressBarThread([this, barIdx]
            {
                while (m_progressBarManager[barIdx].is_completed() == false)
                {
                    m_progressBarManager[barIdx].tick();
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            });

            imagesList = func(wd, imagesList);
            m_progressBarManager[barIdx].set_option(indicators::option::Lead{"·"});
            m_progressBarManager[barIdx].mark_as_completed();

            if (previousWorkingDir)
                m_fileManager.remove(*previousWorkingDir);

            previousWorkingDir = wd;
        };

        for(const auto& op: m_ops)
        {
            execute(op);

            steps--;
            if (steps == 0)
                break;
        }

        for(const auto& op: m_postOps)
            execute(op);

        return imagesList;
    }

private:
    using Op = std::tuple<std::string, Operation, std::string>;
    std::vector<Op> m_ops;
    std::vector<Op> m_postOps;
    Utils::WorkingDir m_wd;
    indicators::DynamicProgress<indicators::IndeterminateProgressBar>& m_progressBarManager;
    const IFileManager& m_fileManager;
    const size_t m_maxSteps;
};
