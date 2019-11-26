cmake_minimum_required(VERSION 3.9)
project(DOT)

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
option(LIBIGL_WITH_TETGEN            "Use Tetgen"         ON)
option(LIBIGL_WITH_TRIANGLE          "Use Triangle"       ON)
option(LIBIGL_WITH_VIEWER            "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_XML               "Use XML"            OFF)

find_package(LIBIGL REQUIRED QUIET)


# tbb
include(tbb/cmake/TBBGet.cmake)
tbb_get(TBB_ROOT tbb_root CONFIG_DIR TBB_DIR)
find_package(TBB REQUIRED tbb)


# project source files
file(GLOB SRCFILES
  "src/*.cpp" 
  "src/Energy/*.cpp" 
  "src/Energy/Physics_Elasticity/*.cpp"
  "src/Utils/*.cpp"
  "src/LinSysSolver/EigenLibSolver.cpp"
  "src/LinSysSolver/CHOLMODSolver.cpp"
  "src/TimeStepper/*.cpp"
  "src/Utils/SVD_EFTYCHIOS/PTHREAD_QUEUE.cpp"
  "src/Utils/SVD_EFTYCHIOS/Singular_Value_Decomposition_Helper.cpp"
  "src/Utils/SVD_EFTYCHIOS/Singular_Value_Decomposition_Preamble.hpp"
  "src/Utils/SVD_EFTYCHIOS/Singular_Value_Decomposition_Jacobi_Conjugation_Kernel.hpp"
  "src/Utils/SVD_EFTYCHIOS/Singular_Value_Decomposition_Givens_QR_Factorization_Kernel.hpp"
  "src/Utils/SVD_EFTYCHIOS/Singular_Value_Decomposition_Main_Kernel_Body.hpp"
  "src/Utils/SVD_EFTYCHIOS/Singular_Value_Decomposition_Kernel_Declarations.hpp"
  "src/Utils/SIMD_DOUBLE_MACROS.hpp"
)
include_directories(
  "src/"
  "src/Energy"
  "src/Energy/Physics_Elasticity"
  "src/Utils"
  "src/Utils/SVD"
  "src/LinSysSolver"
  "src/TimeStepper"
  "SuiteSparse/include"
  "src/Utils/SVD_EFTYCHIOS"
#  "/home/minchen/intel/compilers_and_libraries_2019.1.144/linux/mkl/include"
)


ADD_DEFINITIONS(-DUSE_AVX_IMPLEMENTATION)
add_compile_options(-mavx2)
add_compile_options(-mfma)
add_compile_options(-pthread)


link_directories(${CMAKE_SOURCE_DIR}/SuiteSparse/lib)

add_executable(${PROJECT_NAME}_bin ${SRCFILES})

target_link_libraries(${PROJECT_NAME}_bin igl::core igl::opengl_glfw igl::triangle igl::png igl::tetgen TBB::tbb cholmod metis amd camd ccolamd colamd
#        mkl_intel_lp64
        )
