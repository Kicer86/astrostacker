
module;

#include <filesystem>
#include <functional>
#include <span>
#include <vector>

#include <indicators/dynamic_progress.hpp>
#include <indicators/indeterminate_progress_bar.hpp>

export module execution_plan_builder;
import utils;


export using ImagesList = std::vector<std::filesystem::path>;
export using ImagesView = std::span<const std::filesystem::path>;
export using Operation = std::function<ImagesList(ImagesView)>;

export class ExecutionPlanBuilder
{
public:
    ExecutionPlanBuilder(const WorkingDir& wd, indicators::DynamicProgress<indicators::IndeterminateProgressBar>& progressBarManager)
        : m_wd(wd)
        , m_progressBarManager(progressBarManager)
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
            {
                auto bar = std::make_unique<indicators::IndeterminateProgressBar>(
                    //indicators::option::BarWidth{40},
                    indicators::option::Start{"["},
                    indicators::option::Fill{"·"},
                    indicators::option::Lead{"<==>"},
                    indicators::option::End{"]"},
                    indicators::option::PostfixText{*op.first}
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

                imagesList = op.second(imagesList);
                m_progressBarManager[barIdx].set_option(indicators::option::Lead{"·"});
                m_progressBarManager[barIdx].mark_as_completed();
            }
            else
                imagesList = op.second(imagesList);

        return imagesList;
    }

private:
    std::vector<std::pair<std::optional<std::string>, Operation>> m_ops;
    WorkingDir m_wd;
    indicators::DynamicProgress<indicators::IndeterminateProgressBar>& m_progressBarManager;
};
