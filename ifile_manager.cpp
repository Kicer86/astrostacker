
module;

#include <filesystem>

export module ifile_manager;


export struct IFileManager
{
    virtual ~IFileManager() = default;

    virtual void remove(std::span<const std::filesystem::path>) = 0;
};
