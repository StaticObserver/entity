#ifndef IO_CARGS_H
#define IO_CARGS_H

#include <string_view>
#include <string>
#include <vector>

namespace ntt {
class CommandLineArguments {
private:
  bool _initialized = false;
  std::vector<std::string_view> _args;

public:
  void readCommandLineArguments(int argc, char *argv[]);
  [[nodiscard]] auto getArgument(std::string_view key, std::string_view def) -> std::string_view;
  [[nodiscard]] auto getArgument(std::string_view key) -> std::string_view;
  auto isSpecified(std::string_view key) -> bool;
};
}

#endif
