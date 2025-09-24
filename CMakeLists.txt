cmake_minimum_required(VERSION 3.20)

project(test_cuda LANGUAGES CUDA C CXX)
set(CMAKE_CUDA_ARCHITECTURES 70)

set(SRCS check_cuda.cu)

foreach(SRC_FILE IN LISTS SRCS)
  message("Processing source file: ${SRC_FILE}")

  # 获取文件名（不带扩展名）
  get_filename_component(EXE_NAME ${SRC_FILE} NAME_WE)

  # 创建可执行文件
  add_executable(${EXE_NAME} ${SRC_FILE})

  # 添加 CUDA include 路径
  #target_include_directories(${EXE_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

  #find_library(CUDART_LIBRARY cudart HINTS /usr/local/cuda/lib64)
  #target_link_libraries(${EXE_NAME} PRIVATE ${CUDART_LIBRARY})

  # 可选：设置 CUDA 标准
  set_target_properties(${EXE_NAME} PROPERTIES CUDA_STANDARD 17
                                               CUDA_STANDARD_REQUIRED ON)
endforeach()
