cmake_minimum_required(VERSION 3.16)
project(pinocchio_example)

set(CMAKE_CXX_STANDARD 17)

find_package(glfw3 REQUIRED)
find_package(jsoncpp REQUIRED)
find_package(mujoco REQUIRED)
find_package(pinocchio REQUIRED)

find_package(unitree_sdk2 REQUIRED)

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
    # ${CMAKE_CURRENT_SOURCE_DIR}/src/DiscreteLIPDynamics.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/src/DoubleSupportConfiguration.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/src/FootstepPlanElement.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/src/ISMPC.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/src/JsonConverter.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/src/LIPSimulator.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/src/LIPState.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/JointCommand.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/JointState.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RobotState.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SE3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/StateEstimator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/StateFiltering.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/src/WalkingData.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/WalkingManager.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/src/WalkingState.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/WholeBodyController.cpp
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
  # jsoncpp_lib
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



add_executable(g1_controller
  g1_controller.cpp
)
target_link_libraries(g1_controller
  unitree_sdk2
  pinocchio::pinocchio
  WalkingControllerLibrary
)
# specify include directories
target_include_directories(g1_controller
  PRIVATE
    ${CMAKE_SOURCE_DIR}/unitree_sdk2/example/g1/low_level
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)







target_compile_definitions(
    WalkingControllerLibrary
    PUBLIC
    COAL_DISABLE_HPP_FCL_WARNINGS
)

target_compile_definitions(
    main
    PUBLIC
    COAL_DISABLE_HPP_FCL_WARNINGS
)