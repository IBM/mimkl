#include <iostream>
#include <mimkl/utilities.hpp>
#include <spdlog/spdlog.h>
#include <string>

namespace mimkl
{
namespace utilities
{

std::string demangled(std::string const &sym)
{
    std::unique_ptr<char, void (*)(void *)> name{
    abi::__cxa_demangle(sym.c_str(), nullptr, nullptr, nullptr), std::free};
    return {name.get()};
}

std::shared_ptr<spdlog::logger> logger_checkin(std::string name)
{
    try
    {
        return spdlog::stdout_color_mt(
        name); // TODO see pymimkl wrapper (check for nullptr)
    }
    catch (const spdlog::spdlog_ex &ex)
    {
        std::cout << "Log init failed: " << ex.what() << std::endl;
        return spdlog::get(name);
    }
}

} // namespace utilities
} // namespace mimkl
