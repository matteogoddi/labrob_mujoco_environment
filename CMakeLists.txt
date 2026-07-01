cmake_minimum_required(VERSION 3.16)
project(pinocchio_example)

set(CMAKE_CXX_STANDARD 17)

include_directories(/usr/local/include/ddscxx)

find_package(glfw3 REQUIRED)
find_package(jsoncpp REQUIRED)

find_package(pinocchio REQUIRED)
find_package(unitree_sdk2 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(mujoco REQUIRED)

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()

# Manually set INCLUDE_DIRS and LIBRARIES since not handled in blasfeo and hpipm
set(blasfeo_INCLUDE_DIRS /opt/blasfeo/include)
set(hpipm_INCLUDE_DIRS /opt/hpipm/include)
set(blasfeo_LIBRARIES /opt/blasfeo/lib/libblasfeo.a)
set(hpipm_LIBRARIES /opt/hpipm/lib/libhpipm.a)

add_library(
  WalkingControllerLibrary STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/DiscreteLIPDynamics.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/FootstepPlannerCoop.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/HandAdmittanceController.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ISMPC.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/StateEstimator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/StateFiltering.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/WalkingData.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/WalkingManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/WholeBodyController.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/WristForceEstimator.cpp
)

target_include_directories(WalkingControllerLibrary
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${hpipm_INCLUDE_DIRS}
    ${blasfeo_INCLUDE_DIRS}
)
add_definitions(-DCOAL_DISABLE_HPP_FCL_WARNINGS)

target_link_libraries(WalkingControllerLibrary
  pinocchio::pinocchio
  ${hpipm_LIBRARIES}
  ${blasfeo_LIBRARIES}
  jsoncpp_lib
  OpenMP::OpenMP_CXX
  mujoco::mujoco
)

#add_executable(main
#  main.cpp
#  ${CMAKE_CURRENT_SOURCE_DIR}/src/RobotInterface.cpp
#)
#target_include_directories(main
#  PUBLIC
#    ${CMAKE_CURRENT_SOURCE_DIR}/include
#    mujoco::mujoco
#)
#target_link_libraries(main
#  glfw
#  mujoco::mujoco
#  pinocchio::pinocchio
#  WalkingControllerLibrary
#  unitree_sdk2
#)

add_executable(main_sim
  main_sim.cpp
)
target_include_directories(main_sim
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    mujoco::mujoco
)
target_link_libraries(main_sim
  glfw
  mujoco::mujoco
  pinocchio::pinocchio
  WalkingControllerLibrary
)

add_executable(g1
  main_g1.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/RobotInterface.cpp
)
target_include_directories(g1
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    mujoco::mujoco
)
target_link_libraries(g1
  glfw
  mujoco::mujoco
  pinocchio::pinocchio
  WalkingControllerLibrary
  unitree_sdk2
)