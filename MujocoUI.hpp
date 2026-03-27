
// GLFW
#include <GLFW/glfw3.h>

// Mujoco
#include <mujoco/mujoco.h>

namespace labrob {
class MujocoUI {
 public:
  ~MujocoUI() {
    // close GLFW, free visualization storage
    glfwTerminate();
    mjv_freeScene(&scn_);
    mjr_freeContext(&con_);

    delete mujoco_ui_ptr_;
  }

  static MujocoUI* getInstance() {
    return mujoco_ui_ptr_;
  }

  static MujocoUI* getInstance(mjModel* model_ptr, mjData* data_ptr) {
    if (mujoco_ui_ptr_) {
      return mujoco_ui_ptr_;
    } else {
      mujoco_ui_ptr_ = new MujocoUI();
      mujoco_ui_ptr_->init(model_ptr, data_ptr);
      return mujoco_ui_ptr_;
    }
  }

  void setExternalWristWrenches(
      const mjtNum left_point_world[3],
      const mjtNum left_force_world[3],
      const mjtNum left_torque_world[3],
      bool left_enabled,
      const mjtNum right_point_world[3],
      const mjtNum right_force_world[3],
      const mjtNum right_torque_world[3],
      bool right_enabled
  ) {
    for (int i = 0; i < 3; ++i) {
      left_force_point_world_[i] = left_point_world[i];
      left_force_world_[i] = left_force_world[i];
      left_torque_world_[i] = left_torque_world[i];
      right_force_point_world_[i] = right_point_world[i];
      right_force_world_[i] = right_force_world[i];
      right_torque_world_[i] = right_torque_world[i];
    }
    left_force_enabled_ = left_enabled;
    right_force_enabled_ = right_enabled;
  }

  void render() {
    // get framebuffer viewport
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window_, &viewport.width, &viewport.height);

    // update scene and render
    mjv_updateScene(model_ptr_, data_ptr_, &opt_, NULL, &cam_, mjCAT_ALL, &scn_);

    const auto add_vector_arrow = [this](
        const mjtNum point_world[3],
        const mjtNum vector_world[3],
        const mjtNum vector_visual_scale,
        const mjtNum arrow_radius,
        const float rgba[4]
    ) {
      const mjtNum vector_norm = mju_norm3(vector_world);
      if (vector_norm <= 1e-9 || scn_.ngeom >= scn_.maxgeom) {
        return;
      }

      mjvGeom* geom = scn_.geoms + scn_.ngeom;
      mjv_initGeom(
          geom,
          mjGEOM_ARROW,
          nullptr,
          nullptr,
          nullptr,
          rgba
      );

      mjtNum arrow_end[3] = {
          point_world[0] + vector_visual_scale * vector_world[0],
          point_world[1] + vector_visual_scale * vector_world[1],
          point_world[2] + vector_visual_scale * vector_world[2]
      };
        mjv_connector(geom, mjGEOM_ARROW, arrow_radius, point_world, arrow_end);
      scn_.ngeom += 1;
    };

    static const float left_force_rgba[4] = {0.1f, 0.6f, 1.0f, 1.0f};
    static const float right_force_rgba[4] = {1.0f, 0.4f, 0.1f, 1.0f};
    static const float left_torque_rgba[4] = {0.6f, 0.2f, 1.0f, 1.0f};
    static const float right_torque_rgba[4] = {1.0f, 0.2f, 0.8f, 1.0f};
    if (left_force_enabled_) {
      add_vector_arrow(left_force_point_world_, left_force_world_, 0.05, 0.03, left_force_rgba);
      add_vector_arrow(left_force_point_world_, left_torque_world_, 0.15, 0.02, left_torque_rgba);
    }
    if (right_force_enabled_) {
      add_vector_arrow(right_force_point_world_, right_force_world_, 0.05, 0.03, right_force_rgba);
      add_vector_arrow(right_force_point_world_, right_torque_world_, 0.15, 0.02, right_torque_rgba);
    }

    mjr_render(viewport, &scn_, &con_);

    // swap OpenGL buffers (blocking call due to v-sync)
    glfwSwapBuffers(window_);

    // process pending GUI events, call GLFW callbacks
    glfwPollEvents();
  }

  bool windowShouldClose() {
    auto& mujoco_ui = *mujoco_ui_ptr_;
    return glfwWindowShouldClose(mujoco_ui.window_);
  }

  // mouse button callback
  void onMouseButton(GLFWwindow* window, int button, int act, int mods) {
    auto& mujoco_ui = *mujoco_ui_ptr_;

    // update button state
    button_left_ = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    button_middle_ = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right_ = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(
        window,
        &mujoco_ui.lastx_,
        &mujoco_ui.lasty_
    );
  }

  // mouse move callback
  void onMouseMove(GLFWwindow* window, double xpos, double ypos) {
    auto& mujoco_ui = *mujoco_ui_ptr_;

    // no buttons down: nothing to do
    if (!mujoco_ui.button_left_ &&
        !mujoco_ui.button_middle_ &&
        !mujoco_ui.button_right_) {
      return;
    }

    // compute mouse displacement, save
    double dx = xpos - mujoco_ui.lastx_;
    double dy = ypos - mujoco_ui.lasty_;
    mujoco_ui.lastx_ = xpos;
    mujoco_ui.lasty_ = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if (button_right_) {
      action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    } else if (button_left_) {
      action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    } else {
      action = mjMOUSE_ZOOM;
    }

    // move camera
    mjv_moveCamera(
        mujoco_ui.model_ptr_,
        action,
        dx / height,
        dy / height,
        &mujoco_ui.scn_,
        &mujoco_ui.cam_
    );
  }

  void onScroll(GLFWwindow* window, double xoffset, double yoffset) {
    auto& mujoco_ui = *mujoco_ui_ptr_;

    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(
        mujoco_ui.model_ptr_,
        mjMOUSE_ZOOM,
        0,
        -0.05 * yoffset,
        &mujoco_ui.scn_,
        &mujoco_ui.cam_
    );
  }

 protected:
  MujocoUI() = default;

  void init(mjModel* model_ptr, mjData* data_ptr) {
    auto& mujoco_ui = *mujoco_ui_ptr_;

    mujoco_ui.model_ptr_ = model_ptr;
    mujoco_ui.data_ptr_ = data_ptr;
    mujoco_ui.button_left_ = false;
    mujoco_ui.button_middle_ = false;
    mujoco_ui.button_right_ = false;
    mujoco_ui.lastx_ = 0.0;
    mujoco_ui.lasty_ = 0.0;
    mju_zero3(mujoco_ui.left_force_point_world_);
    mju_zero3(mujoco_ui.left_force_world_);
    mju_zero3(mujoco_ui.left_torque_world_);
    mju_zero3(mujoco_ui.right_force_point_world_);
    mju_zero3(mujoco_ui.right_force_world_);
    mju_zero3(mujoco_ui.right_torque_world_);
    mujoco_ui.left_force_enabled_ = false;
    mujoco_ui.right_force_enabled_ = false;

    // init GLFW, create window, make OpenGL context current, request v-sync
    glfwInit();
    window_ = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam_);
    //mjv_defaultPerturb(&pert);
    mjv_defaultOption(&opt_);
    mjr_defaultContext(&con_);

    // Visualize contact points and contact forces
    opt_.flags[mjVIS_CONTACTPOINT] = true;
    opt_.flags[mjVIS_CONTACTFORCE] = true;
    opt_.flags[mjVIS_TRANSPARENT] = true;

    cam_.distance = 4.0;

    // create scene and context
    mjv_makeScene(model_ptr_, &scn_, 5000);
    mjr_makeContext(model_ptr_, &con_, mjFONTSCALE_100);

    // install GLFW mouse callbacks
    glfwSetCursorPosCallback(
        window_,
        [](GLFWwindow* window, double xpos, double ypos) {
          MujocoUI::getInstance()->onMouseMove(window, xpos, ypos);
        }
    );
    glfwSetMouseButtonCallback(
        window_,
        [](GLFWwindow* window, int button, int act, int mods) {
          MujocoUI::getInstance()->onMouseButton(window, button, act, mods);
        }
    );
    glfwSetScrollCallback(
        window_,
        [](GLFWwindow* window, double xoffset, double yoffset) {
          MujocoUI::getInstance()->onScroll(window, xoffset, yoffset);
        }
    );
  }

  static MujocoUI* mujoco_ui_ptr_;

 private:

  MujocoUI(MujocoUI& rhs) = delete;
  MujocoUI operator=(const MujocoUI& rhs) = delete;

  // Mujoco model:
  mjModel* model_ptr_;
  mjData* data_ptr_;

  // Mujoco visualization:
  mjvCamera cam_;                      // abstract camera
  mjvOption opt_;                      // visualization options
  mjvScene scn_;                       // abstract scene
  mjrContext con_;                     // custom GPU context

  GLFWwindow* window_;

  // Mouse interaction:
  bool button_left_;
  bool button_middle_;
  bool button_right_;
  double lastx_;
  double lasty_;

  mjtNum left_force_point_world_[3];
  mjtNum left_force_world_[3];
  mjtNum left_torque_world_[3];
  mjtNum right_force_point_world_[3];
  mjtNum right_force_world_[3];
  mjtNum right_torque_world_[3];
  bool left_force_enabled_;
  bool right_force_enabled_;

};

MujocoUI* MujocoUI::mujoco_ui_ptr_ = nullptr;
} // end namespace labrob