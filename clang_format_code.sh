#!/bin/bash

# format sources
find . -path ./external -prune -o -name "*.cpp" -exec clang-format -i {} \;
# format heades
find . -path ./external -prune -o -name "*.hpp" -exec clang-format -i {} \;
