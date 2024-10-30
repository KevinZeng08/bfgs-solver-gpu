cmake_minimum_required(VERSION 3.16)

project(BFGSProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

# 添加源文件
set(CPU_SOURCES bfgs-cpp.cpp)
set(CUDA_SOURCES bfgs-cu.cu)

# 添加可执行文件 - CPU版本
add_executable(BFGSCPU ${CPU_SOURCES})

# 链接OpenMP库
target_link_libraries(BFGSCPU PUBLIC OpenMP::OpenMP_CXX)

# 设置CPU编译选项
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(BFGSCPU PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -g>
    )
else(
    target_compile_options(BFGSCPU PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
    )
)
endif()

# 添加可执行文件 - CUDA版本
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bfgs-cu.cu")
    find_package(CUDA REQUIRED)
    add_executable(BFGSCUDA ${CUDA_SOURCES})

    # 链接OpenMP库
    target_link_libraries(BFGSCUDA PUBLIC OpenMP::OpenMP_CXX)
    
    # 设置CUDA编译选项
    set_target_properties(BFGSCUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(BFGSCUDA PROPERTIES
        CUDA_ARCHITECTURES "86;80;75;72")
    # debug
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(BFGSCUDA PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>: -G -g>
        )
    else()
        target_compile_options(BFGSCUDA PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>: -O3>
        )
    endif()
endif()

# 为方便使用，可以添加别名目标
add_custom_target(cpu
    COMMAND ${CMAKE_COMMAND} --build . --target BFGSCPU
    COMMENT "Building CPU version"
)

add_custom_target(cuda
    COMMAND ${CMAKE_COMMAND} --build . --target BFGSCUDA
    COMMENT "Building CUDA version"
    DEPENDS BFGSCUDA
)
