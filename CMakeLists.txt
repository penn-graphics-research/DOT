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


# project source files
file(GLOB SRCFILES
  "FracCuts/*.cpp" 
  "FracCuts/Energy/*.cpp" 
  "FracCuts/Energy/Physics_Elasticity/*.cpp"
  "FracCuts/Energy/Geometry_Isometry/*.cpp"
  "FracCuts/Energy/Connectivity/*.cpp"
  "FracCuts/Utils/*.cpp"
  "FracCuts/LinSysSolver/EigenLibSolver.cpp"
  "FracCuts/TimeStepper/*.cpp"
)
include_directories(
  "FracCuts"
  "FracCuts/Energy"
  "FracCuts/Energy/Physics_Elasticity"
  "FracCuts/Energy/Geometry_Isometry"
  "FracCuts/Energy/Connectivity"
  "FracCuts/Utils"
  "FracCuts/LinSysSolver"
  "FracCuts/TimeStepper"
)


add_executable(${PROJECT_NAME}_bin ${SRCFILES})

target_link_libraries(${PROJECT_NAME}_bin igl::core igl::opengl_glfw igl::triangle igl::png)
