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

set(CASADI_ROOT "" CACHE PATH "Root directory of local CasADi install (contains include/ and libcasadi.so)")
set(CASADI_ABI_OVERRIDE "auto" CACHE STRING "CasADi ABI override: auto|0|1")
set_property(CACHE CASADI_ABI_OVERRIDE PROPERTY STRINGS auto 0 1)

find_path(CASADI_INCLUDE_DIR
  NAMES casadi/casadi.hpp
  HINTS
    ${CASADI_ROOT}/include
    $ENV{HOME}/.local/lib/python3.8/site-packages/casadi/include
    /usr/local/include
    /usr/include
)

find_library(CASADI_LIBRARY
  NAMES casadi libcasadi
  HINTS
    ${CASADI_ROOT}
    ${CASADI_ROOT}/lib
    $ENV{HOME}/.local/lib/python3.8/site-packages/casadi
    /usr/local/lib
    /usr/lib
)

message(STATUS "CASADI_INCLUDE_DIR = ${CASADI_INCLUDE_DIR}")
message(STATUS "CASADI_LIBRARY     = ${CASADI_LIBRARY}")

if(NOT CASADI_INCLUDE_DIR OR NOT CASADI_LIBRARY)
  message(FATAL_ERROR "CasADi not found. Please install CasADi C++ library or set CASADI_INCLUDE_DIR and CASADI_LIBRARY.")
endif()

# ABI alignment for CasADi (CMake-only workaround for local CasADi binary mismatch)
set(CASADI_NEEDS_OLD_ABI OFF)
if(CASADI_ABI_OVERRIDE STREQUAL "0")
  set(CASADI_NEEDS_OLD_ABI ON)
elseif(CASADI_ABI_OVERRIDE STREQUAL "1")
  set(CASADI_NEEDS_OLD_ABI OFF)
else()
  execute_process(
    COMMAND bash -lc "nm -D '${CASADI_LIBRARY}' | c++filt | grep -m1 'casadi::conic(std::__cxx11::basic_string' >/dev/null && echo 1 || echo 0"
    OUTPUT_VARIABLE CASADI_HAS_CXX11_STRING_API
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(CASADI_HAS_CXX11_STRING_API STREQUAL "1")
    set(CASADI_NEEDS_OLD_ABI OFF)
  else()
    set(CASADI_NEEDS_OLD_ABI ON)
  endif()
endif()

if(CASADI_NEEDS_OLD_ABI)
  message(WARNING "CasADi appears to be built with old libstdc++ ABI; applying _GLIBCXX_USE_CXX11_ABI=0 to CasADi-dependent sources.")
else()
  message(STATUS "CasADi appears compatible with C++11 ABI (_GLIBCXX_USE_CXX11_ABI=1 default).")
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

if(CASADI_NEEDS_OLD_ABI)
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ComplianceReferenceGenerator.cpp
    PROPERTIES COMPILE_DEFINITIONS _GLIBCXX_USE_CXX11_ABI=0
  )
endif()

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
  ${CASADI_LIBRARY}
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
  ${CASADI_LIBRARY}
)

add_executable(validate_torso_compliance_qp
  tools/validate_torso_compliance_qp.cpp
)
if(CASADI_NEEDS_OLD_ABI)
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/validate_torso_compliance_qp.cpp
    PROPERTIES COMPILE_DEFINITIONS _GLIBCXX_USE_CXX11_ABI=0
  )
endif()
target_include_directories(validate_torso_compliance_qp
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(validate_torso_compliance_qp
  WalkingControllerLibrary
  ${CASADI_LIBRARY}
)
