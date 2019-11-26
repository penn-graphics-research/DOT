cmake_minimum_required(VERSION 3.1)

project(cork)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

add_library(cork STATIC
  src/file_formats/files.cpp
  src/file_formats/ifs.cpp
  src/file_formats/off.cpp
  src/isct/empty3d.cpp
  src/isct/quantization.cpp
  src/isct/triangle.c
  src/util/log.cpp
  src/util/timer.cpp
  src/cork.cpp)

add_executable(cork-bin src/main.cpp)

set_target_properties(cork PROPERTIES CXX_STANDARD 11)
set_target_properties(cork PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(cork-bin PROPERTIES CXX_STANDARD 11)
set_target_properties(cork-bin PROPERTIES CXX_STANDARD_REQUIRED ON)

set(INCLUDEDIRS
  src/mesh
  src/accel/
  src/file_formats/
  src/isct/
  src/math/
  src/mesh/
  src/rawmesh/
  src/util/
)

target_include_directories(cork PRIVATE ${INCLUDEDIRS})
target_include_directories(cork-bin PRIVATE ${INCLUDEDIRS})

set_target_properties(cork-bin PROPERTIES OUTPUT_NAME cork)

find_package(GMP REQUIRED)

target_link_libraries(cork PUBLIC ${GMP_LIBRARIES})
target_include_directories(cork PUBLIC ${GMP_INCLUDES})
target_compile_definitions(cork PRIVATE
  -DNO_TIMER -DREDUCED -DCDT_ONLY -DTRILIBRARY -DANSI_DECLARATORS)

target_link_libraries(cork-bin PRIVATE cork)
