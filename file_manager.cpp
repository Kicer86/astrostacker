
module;

#include <filesystem>
#include <set>
#include <span>

export module file_manager;
import ifile_manager;
import utils;


export class FileManager: public IFileManager
{
public:
    FileManager(bool doRemoval) : m_remove(doRemoval) {}

    void remove(const Utils::WorkingDir& wd) const override
    {
        if (m_remove == false)
            return;

        std::filesystem::remove_all(wd.path());
    }

private:
    const bool m_remove;
};
