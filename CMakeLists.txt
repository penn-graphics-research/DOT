cmake_minimum_required(VERSION 3.9)
project(OptCuts_dynamic)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


# libigl
option(LIBIGL_USE_STATIC_LIBRARY "Use libigl as static library" OFF)
option(LIBIGL_WITH_ANTTWEAKBAR       "Use AntTweakBar"    OFF)
option(LIBIGL_WITH_CGAL              "Use CGAL"           OFF)
option(LIBIGL_WITH_COMISO            "Use CoMiso"         OFF)
option(LIBIGL_WITH_CORK              "Use Cork"           OFF)
option(LIBIGL_WITH_EMBREE            "Use Embree"         OFF)
option(LIBIGL_WITH_LIM               "Use LIM"            OFF)
option(LIBIGL_WITH_MATLAB            "Use Matlab"         OFF)
option(LIBIGL_WITH_MOSEK             "Use MOSEK"          OFF)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"          OFF)
option(LIBIGL_WITH_PNG               "Use PNG"            ON)
option(LIBIGL_WITH_PYTHON            "Use Python"         OFF)
option(LIBIGL_WITH_TETGEN            "Use Tetgen"         OFF)
option(LIBIGL_WITH_TRIANGLE          "Use Triangle"       ON)
option(LIBIGL_WITH_VIEWER            "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_XML               "Use XML"            OFF)

find_package(LIBIGL REQUIRED QUIET)


# Check to see if PARDISO is available
find_library(PARDISO
  pardiso500-INTEL1301-X86-64
  pardiso500-GNU461-X86-64
  pardiso500-GNU472-X86-64
  pardiso500-GNU481-X86-64
  pardiso500-MPI-INTEL1301-X86-64
  pardiso500-MPI-GNU450-X86-64
  pardiso500-MPI-GNU461-X86-64
  pardiso500-MPI-GNU463-X86-64
  pardiso500-MPI-GNU472-X86-64
  pardiso500-WIN-X86-64
  pardiso500-MACOS-X86-64
  PATHS ${CMAKE_CURRENT_SOURCE_DIR}/ext/pardiso
)


# TBB
include(tbb/cmake/TBBGet.cmake)
tbb_get(TBB_ROOT tbb_root CONFIG_DIR TBB_DIR)
find_package(TBB REQUIRED tbb)


# CHOLMOD
find_library(SUITESPARSE
  libcholmod.3.0.12
  PATHS ${CMAKE_CURRENT_SOURCE_DIR}/ext/SUITESPARSE
)
include_directories(
  "/Users/mincli/Documents/SuiteSparse/include"
)

# project source files
file(GLOB SRCFILES 
  "FracCuts/*.cpp" 
  "FracCuts/Energy/*.cpp" 
  "FracCuts/Energy/Physics_Elasticity/*.cpp"
  "FracCuts/Energy/Geometry_Isometry/*.cpp"
  "FracCuts/Energy/Connectivity/*.cpp"
  "FracCuts/Utils/*.cpp"
)
include_directories(
  "FracCuts"
  "FracCuts/Energy"
  "FracCuts/Utils"
)


add_executable(${PROJECT_NAME}_bin ${SRCFILES})

target_link_libraries(${PROJECT_NAME}_bin igl::core igl::opengl_glfw igl::triangle igl::png ${PARDISO} TBB::tbb ${SUITESPARSE})
