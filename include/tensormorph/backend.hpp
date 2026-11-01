
#include <string_view>
namespace tensormorph{
    enum class Backend{
        CPU
    };

    [[nodiscard]]
    constexpr std::string_view backend_name(Backend backend) noexcept{
        switch (backend) {
            case tensormorph::Backend::CPU:
                return "CPU";
        }

        return "Unknown";
    }
}