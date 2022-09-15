cmake_minimum_required(VERSION 3.10)

project("yolov7 tflite")

if("${ARCH}" STREQUAL "aarch64")
  message("Build ${ARCH}")
  set(TFLITE_DIR tflite-aarch64)
  set(ONNX_DIR onnx-aarch64)
else()
  message("Build x86_64")
  set(TFLITE_DIR tflite)
  set(ONNX_DIR onnx)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)
include_directories(${OPENCV_INCLUDE_DIRS})

add_library(tflite STATIC IMPORTED)
set_target_properties(tflite PROPERTIES
  IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/${TFLITE_DIR}/libtensorflow-lite.a
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/${TFLITE_DIR}
)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/${TFLITE_DIR}/lib/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/${TFLITE_DIR}/include/)

set(TFLITE_LIB
  tflite
  XNNPACK
  cpuinfo
  pthreadpool
  clog
  fft2d_fftsg
  fft2d_fftsg2d
  flatbuffers
  farmhash
  ruy_allocator
  ruy_apply_multiplier
  ruy_context
  ruy_context_get_ctx
  ruy_cpuinfo
  ruy_ctx
  ruy_denormal
  ruy_frontend
  ruy_have_built_path_for_avx
  ruy_have_built_path_for_avx2_fma
  ruy_have_built_path_for_avx512
  ruy_kernel_arm
  ruy_kernel_avx
  ruy_kernel_avx2_fma
  ruy_kernel_avx512
  ruy_pack_arm
  ruy_pack_avx
  ruy_pack_avx2_fma
  ruy_pack_avx512
  ruy_prepacked_cache
  ruy_prepare_packed_matrices
  ruy_profiler_instrumentation
  ruy_profiler_profiler
  ruy_system_aligned_alloc
  ruy_thread_pool
  ruy_trmul
  ruy_blocking_counter
  ruy_block_map
  ruy_tune
  ruy_wait
)

add_executable(main
  opencv_yolov7_tflite.cc
)
target_link_libraries(main
  ${OPENCV_STATIC_LDFLAGS}
  ${TFLITE_LIB}
)

add_executable(bench
  benchmark.cc
)
target_link_libraries(bench
  ${OPENCV_STATIC_LDFLAGS}
  ${TFLITE_LIB}
)
target_link_libraries(bench
  ${OPENCV_STATIC_LDFLAGS}
  ${TFLITE_LIB}
)

add_library(onnx STATIC IMPORTED)
set_target_properties(onnx PROPERTIES
  IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/${ONNX_DIR}/lib/libonnxruntime.so
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/${ONNX_DIR}/include
)
add_executable(onnx_main
  opencv_yolov7_onnx.cc
)
target_link_libraries(onnx_main
  ${OPENCV_STATIC_LDFLAGS}
  onnx
)


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test.jpg ./test.jpg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/best.tflite ./best.tflite COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/best-320.onnx ./best-320.onnx COPYONLY)
