# The official 3.4.0 Linux tarball ships headers and the shared library.
if(NOT TARGET mujoco::mujoco)
  get_filename_component(_mujoco_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
  add_library(mujoco::mujoco SHARED IMPORTED)
  set_target_properties(mujoco::mujoco PROPERTIES
    IMPORTED_LOCATION "${_mujoco_root}/lib/libmujoco.so.3.4.0"
    INTERFACE_INCLUDE_DIRECTORIES "${_mujoco_root}/include"
    INTERFACE_COMPILE_DEFINITIONS mjUSEPLATFORMSIMD)
endif()
