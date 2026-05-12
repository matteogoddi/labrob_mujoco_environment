cmake_minimum_required(VERSION 3.16...3.30)
project(pinocchio_example)

set(CMAKE_CXX_STANDARD 17)

include_directories(/usr/local/include/ddscxx)

find_package(glfw3 REQUIRED)
find_package(jsoncpp REQUIRED)
find_package(mujoco REQUIRED)
find_package(pinocchio REQUIRED)
find_package(unitree_sdk2 REQUIRED)
find_package(OpenMP REQUIRED)

find_path(CASADI_INCLUDE_DIR
  NAMES casadi/casadi.hpp
  HINTS
    $ENV{HOME}/.local/lib/python3.8/site-packages/casadi/include
    /usr/local/include
    /usr/include
)

find_library(CASADI_LIBRARY
  NAMES casadi libcasadi
  HINTS
    $ENV{HOME}/.local/lib/python3.8/site-packages/casadi
    /usr/local/lib
    /usr/lib
)

if(NOT CASADI_INCLUDE_DIR OR NOT CASADI_LIBRARY)
  message(FATAL_ERROR "CasADi not found. Please install CasADi C++ library or set CASADI_INCLUDE_DIR and CASADI_LIBRARY.")
endif()

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()

# Add local blasfeo and hpipm as subdirectories
add_subdirectory(blasfeo)
add_subdirectory(hpipm)

# Set INCLUDE_DIRS and LIBRARIES for local copies
set(blasfeo_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/blasfeo/include)
set(hpipm_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/hpipm/include)
set(blasfeo_LIBRARIES blasfeo)
set(hpipm_LIBRARIES hpipm)

add_library(
  WalkingControllerLibrary STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ComplianceReferenceGenerator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/DiscreteLIPDynamics.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/DoubleSupportConfiguration.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/FootstepPlanElement.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ISMPC.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/JsonConverter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/LIPSimulator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/LIPState.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ResidualEstimator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/EstimateForce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/JointCommand.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/JointState.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RobotState.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SE3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/WalkingData.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/WalkingManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/WalkingState.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/WholeBodyController.cpp
)

target_include_directories(WalkingControllerLibrary
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${hpipm_INCLUDE_DIRS}
    ${blasfeo_INCLUDE_DIRS}
    ${CASADI_INCLUDE_DIR}
)

target_link_libraries(WalkingControllerLibrary
  pinocchio::pinocchio
  ${hpipm_LIBRARIES}
  ${blasfeo_LIBRARIES}
  ${CASADI_LIBRARY}
  jsoncpp_lib
  OpenMP::OpenMP_CXX
)

add_executable(main
  main.cpp
)
target_include_directories(main
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(main
  glfw
  mujoco::mujoco
  pinocchio::pinocchio
  WalkingControllerLibrary
  unitree_sdk2
)

add_executable(lip_simulator_main
lip_simulator_main.cpp
)
target_include_directories(lip_simulator_main
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(lip_simulator_main
  pinocchio::pinocchio
  WalkingControllerLibrary
)