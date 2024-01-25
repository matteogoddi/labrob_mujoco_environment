cmake_minimum_required(VERSION 3.16)
project(pinocchio_example)
find_package(glfw3 REQUIRED)
find_package(mujoco REQUIRED)
find_package(pinocchio REQUIRED)
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
)
