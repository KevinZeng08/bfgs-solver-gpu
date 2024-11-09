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

find_package(CUDA REQUIRED)

function(set_cuda_target_properties target_name)
    set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES "86;80;75;72")
    
    # 添加CUTLASS包含路径
    target_include_directories(${target_name} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cutlass/include
    )

    if(USE_CUTLASS)
        add_definitions(-DUSE_CUTLASS)
    endif()

    # Debug编译选项
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>: -G -g>
        )
    else()
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>: -O3 -march=native -funroll-loops>
        )
    endif()

    # 链接OpenMP库
    target_link_libraries(${target_name} PUBLIC OpenMP::OpenMP_CXX)
endfunction()

# 添加可执行文件 - CUDA版本
add_executable(BFGSCUDA ${CUDA_SOURCES})
set_cuda_target_properties(BFGSCUDA)

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --fmad=false --ftz=false --prec-div=true --prec-sqrt=true")
# remove cutlass warning
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

add_executable(evaluate evaluate.cu)
set_cuda_target_properties(evaluate)

# 为方便使用，可以添加别名目标
add_custom_target(cpu
    COMMAND ${CMAKE_COMMAND} --build . --target BFGSCPU
    COMMENT "Building CPU version"
)

add_custom_target(cuda
    COMMAND ${CMAKE_COMMAND} --build . --target BFGSCUDA
    COMMENT "Building CUDA version"
)

add_custom_target(eval
    COMMAND ${CMAKE_COMMAND} --build . --target evaluate
    COMMENT "Evaluating GEMV"
)
