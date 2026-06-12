
// GLFW
#include <GLFW/glfw3.h>
#include <memory>

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
  }

  static MujocoUI* getInstance() {
    return mujoco_ui_ptr_.get();
  }

  static MujocoUI* getInstance(mjModel* model_ptr, mjData* data_ptr) {
    if (mujoco_ui_ptr_) {
      return mujoco_ui_ptr_.get();
    } else {
      mujoco_ui_ptr_.reset(new MujocoUI());
      mujoco_ui_ptr_->init(model_ptr, data_ptr);
      return mujoco_ui_ptr_.get();
    }
  }

  void render() {
    // get framebuffer viewport
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window_, &viewport.width, &viewport.height);

    // update scene and render
    mjv_updateScene(model_ptr_, data_ptr_, &opt_, NULL, &cam_, mjCAT_ALL, &scn_);
    mjr_render(viewport, &scn_, &con_);

    // swap OpenGL buffers (blocking call due to v-sync)
    glfwSwapBuffers(window_);

    // process pending GUI events, call GLFW callbacks
    glfwPollEvents();
  }

  bool windowShouldClose() {
    return glfwWindowShouldClose(window_);
  }

  // mouse button callback
  void onMouseButton(GLFWwindow* window, int button, int act, int mods) {
    // update button state
    button_left_   = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)   == GLFW_PRESS);
    button_middle_ = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right_  = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)  == GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx_, &lasty_);
  }

  // mouse move callback
  void onMouseMove(GLFWwindow* window, double xpos, double ypos) {
    // no buttons down: nothing to do
    if (!button_left_ && !button_middle_ && !button_right_) {
      return;
    }

    // compute mouse displacement, save
    double dx = xpos - lastx_;
    double dy = ypos - lasty_;
    lastx_ = xpos;
    lasty_ = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)  == GLFW_PRESS ||
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
    mjv_moveCamera(model_ptr_, action, dx / height, dy / height, &scn_, &cam_);
  }

  void onScroll(GLFWwindow* window, double xoffset, double yoffset) {
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(model_ptr_, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn_, &cam_);
  }

 protected:
  MujocoUI() = default;

  void init(mjModel* model_ptr, mjData* data_ptr) {
    model_ptr_     = model_ptr;
    data_ptr_      = data_ptr;
    button_left_   = false;
    button_middle_ = false;
    button_right_  = false;
    lastx_         = 0.0;
    lasty_         = 0.0;

    // init GLFW, create window, make OpenGL context current, request v-sync
    glfwInit();
    window_ = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam_);
    mjv_defaultOption(&opt_);
    mjr_defaultContext(&con_);

    // Visualize contact points and contact forces
    opt_.flags[mjVIS_CONTACTPOINT] = true;
    opt_.flags[mjVIS_CONTACTFORCE] = true;
    opt_.flags[mjVIS_TRANSPARENT]  = true;

    cam_.distance = 4.0;

    // create scene and context
    mjv_makeScene(model_ptr_, &scn_, 1000);
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

  static std::unique_ptr<MujocoUI> mujoco_ui_ptr_;

 private:
  MujocoUI(const MujocoUI& rhs) = delete;
  MujocoUI& operator=(const MujocoUI& rhs) = delete;

  // Mujoco model:
  mjModel* model_ptr_;
  mjData* data_ptr_;

  // Mujoco visualization:
  mjvCamera  cam_;   // abstract camera
  mjvOption  opt_;   // visualization options
  mjvScene   scn_;   // abstract scene
  mjrContext con_;   // custom GPU context

  GLFWwindow* window_;

  // Mouse interaction:
  bool   button_left_;
  bool   button_middle_;
  bool   button_right_;
  double lastx_;
  double lasty_;
};

std::unique_ptr<MujocoUI> MujocoUI::mujoco_ui_ptr_;
} // end namespace labrob