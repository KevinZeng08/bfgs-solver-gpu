# 使用GPU优化BFGS求解器

## How to Use

Environment: Ubuntu 20.04, CUDA 12.1

Install CMake and CUDA

```bash
# install cmake from Kitware
# 添加Kitware的APT仓库密钥
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null

# 添加Kitware的APT仓库
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"

# 更新并安装
sudo apt update
sudo apt install cmake
```

Clone this repository and build

```bash
# with submodules
git clone --recurse-submodules https://github.com/KevinZeng08/bfgs-solver-gpu.git
```

```bash
# build target
mkdir build && cd build
cmake -DUSE_CUTLASS=ON ..
make # optional targets: cpu,cuda,eval
./BFGSCPU ../data/bfgs-large.dat
./BFGSCUDA ../data/bfgs-large.dat
```
## Results

Environment
- CPU: Intel(R) Xeon(R) Gold 5218R CPU @ 2.10GHz
- GPU: NVIDIA RTX A5000 24GB, CUDA 12.1
- OS: Ubuntu 20.04

**49x** speedup

|  | BFGS-CPU | BFGS-CUDA |
| --- | --- | --- |
| Latency (s) | 57.2 | 1.17 |
