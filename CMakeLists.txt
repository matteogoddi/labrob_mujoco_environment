cmake_minimum_required(VERSION 3.16)
project(pinocchio_example)

find_package(glfw3 REQUIRED)
find_package(mujoco REQUIRED)
find_package(pinocchio REQUIRED)

# Manually set INCLUDE_DIRS and LIBRARIES since not handled in blasfeo and hpipm
set(blasfeo_INCLUDE_DIRS /opt/blasfeo/include)
set(hpipm_INCLUDE_DIRS /opt/hpipm/include)
set(blasfeo_LIBRARIES /opt/blasfeo/lib/libblasfeo.a)
set(hpipm_LIBRARIES /opt/hpipm/lib/libhpipm.a)

add_library(
  WalkingControllerLibrary STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/DoubleSupportConfiguration.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/FootstepPlanElement.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ISMPC.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ISMPCState.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/JointCommand.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/JointState.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RobotState.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SE3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/WalkingData.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/WalkingManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/WalkingState.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/WholeBodyController.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/TorqueWholeBodyController.cpp
)

target_include_directories(WalkingControllerLibrary
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${hpipm_INCLUDE_DIRS}
    ${blasfeo_INCLUDE_DIRS}
)

target_link_libraries(WalkingControllerLibrary
  pinocchio::pinocchio
  ${hpipm_LIBRARIES}
  ${blasfeo_LIBRARIES}
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
)
