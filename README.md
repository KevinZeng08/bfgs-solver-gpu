# 浙江大学GPU计算与应用课程作业

## 任务概述

优化BFGS求解器

## Quickstart

Clone this repository and build

```bash
# with submodules
git clone --recurse-submodules https://github.com/KevinZeng08/bfgs-solver-gpu.git
```

```bash
# build target
mkdir build && cd build
cmake ..
make # optional targets: cpu,cuda,eval
./BFGSCPU ../data/bfgs-large.dat
./BFGSCUDA ../data/bfgs-large.dat
```

## Use cutlass

```bash
cmake .. -DUSE_CUTLASS=ON
make
```

## Results

优化的结果可视化对比

## Methods

使用了哪些优化手段