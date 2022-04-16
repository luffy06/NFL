cmake_minimum_required(VERSION 3.12)
project(KV-Evaluator VERSION 1.0 LANGUAGES CXX)

# specify c/c++ standard
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif(NOT CMAKE_CXX_STANDARD)

list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/lib/libmkl")

set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(LIB_DIR "${PROJECT_SOURCE_DIR}/lib")

# Configuration for ALEX
if (MSVC)
  set(CMAKE_CXX_FLAGS "/O2 /arch:AVX2 /W1 /EHsc")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS "-O3 -xHost -ggdb3 -pthread")
else ()
  set(CMAKE_CXX_FLAGS "-O3 -march=native -ggdb3 -pthread")
endif ()

include_directories(
  ${SRC_DIR}
  ${LIB_DIR}
)

add_executable(format "${SRC_DIR}/format_data.cc")
add_executable(gen "${SRC_DIR}/data_generator.cc")
add_executable(nf_convert "${SRC_DIR}/nf_data_converter.cc")

add_executable(evaluate "${SRC_DIR}/main.cc")
add_executable(evaluate_nf "${SRC_DIR}/evaluate_nf.cc")

add_executable(test_nfl "${SRC_DIR}/test/test_nfl.cc")
add_executable(test_afli "${SRC_DIR}/test/test_afli.cc")
add_executable(test_bnaf "${SRC_DIR}/test/test_numerical_flow.cc")
add_executable(test_lm "${SRC_DIR}/test/test_linear_model.cc")

add_executable(cache_workload "${SRC_DIR}/analysis/cache_workload.cc")
add_executable(aggregation "${SRC_DIR}/analysis/aggregation.cc")
# add_executable(update_index "${SRC_DIR}/analysis/update_index.cc")

find_package(MKL)
if (MKL_FOUND)
  include_directories(${MKL_INCLUDE_DIR})
  target_link_libraries(evaluate ${MKL_LIBRARIES})
  target_link_libraries(evaluate_nf ${MKL_LIBRARIES})
  target_link_libraries(test_bnaf ${MKL_LIBRARIES})
  target_link_libraries(test_lm ${MKL_LIBRARIES})
  target_link_libraries(test_nfl ${MKL_LIBRARIES})
  target_link_libraries(aggregation ${MKL_LIBRARIES})
  # target_link_libraries(update_index ${MKL_LIBRARIES})
else ()
  message(WARNING "MKL libs not found")
endif ()

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else ()
  message(WARNING "OpenMP libs not found")
endif ()
