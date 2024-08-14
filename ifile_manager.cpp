
module;

#include <filesystem>
#include <span>


export module ifile_manager;

import utils;

export struct IFileManager
{
    virtual ~IFileManager() = default;

    virtual void remove(const Utils::WorkingDir& wd) const = 0;
};
