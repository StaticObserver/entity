#!/bin/bash

# 构建项目而不显示警告的脚本

echo "构建项目并抑制所有警告..."

# 方法1：使用cmake配置选项
mkdir -p build_no_warnings
cd build_no_warnings

cmake .. -DSUPPRESS_WARNINGS=ON -DDEBUG=ON
make -j$(nproc)

echo "构建完成！警告已被抑制。"
echo ""
echo "使用方法："
echo "1. 使用脚本: ./build_without_warnings.sh"
echo "2. 手动配置: cmake -DSUPPRESS_WARNINGS=ON -DDEBUG=ON .."
echo "3. 或者设置环境变量: export Entity_SUPPRESS_WARNINGS=ON" 