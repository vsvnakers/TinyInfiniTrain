#!/bin/bash

# 初始化子模块配置
git submodule init

# 添加每个子模块
git submodule add git@github.com:google/glog.git third_party/glog
git submodule add git@github.com:gflags/gflags.git third_party/gflags
git submodule add https://gitlab.com/libeigen/eigen.git third_party/eigen
git submodule add git@github.com:google/googletest.git third_party/googletest
