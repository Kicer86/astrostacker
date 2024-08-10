
module;

#include <filesystem>
#include <set>

export module file_manager;
import ifile_manager;

export class FileManager: public IFileManager
{
public:
    FileManager(bool doRemoval) : m_remove(doRemoval) {}

    void ignore(std::span<const std::filesystem::path> paths)
    {
        for (const auto& path: paths)
            m_ignored.insert(path);
    }

    void remove(std::span<const std::filesystem::path> paths) override
    {
        if (m_remove == false)
            return;

        for (const auto& path: paths)
            if (m_ignored.contains(path) == false)
                std::filesystem::remove(path);
    }

private:
    std::set<std::filesystem::path> m_ignored;
    const bool m_remove;
};
