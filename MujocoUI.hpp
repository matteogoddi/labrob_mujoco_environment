
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

    void addForceArrow(
      const mjtNum* from,      // punto di applicazione [3]
      const mjtNum* force,     // vettore forza [3]
      float scale,             // es. 0.005f (m/N)
      const float* rgba        // colore [4]
  ) {
      if (scn_.ngeom >= scn_.maxgeom) return;

      // Calcola il punto "to" = from + scale * force
      mjtNum to[3] = {
          from[0] + scale * force[0],
          from[1] + scale * force[1],
          from[2] + scale * force[2]
      };

      mjvGeom* g = scn_.geoms + scn_.ngeom;
      mjv_initGeom(g, mjGEOM_ARROW, NULL, NULL, NULL, rgba);
      mjv_connector(g, mjGEOM_ARROW, /*width=*/0.01, from, to);
      scn_.ngeom++;
  }

  struct ForceArrow {
      Eigen::Vector3d from;
      Eigen::Vector3d force;
      float rgba[4];
  };

  // Renders the scene with an arbitrary list of force arrows.
  void renderWithForces(const std::vector<ForceArrow>& arrows, float scale = 0.05f) {
      mjrRect viewport = {0, 0, 0, 0};
      glfwGetFramebufferSize(window_, &viewport.width, &viewport.height);
      mjv_updateScene(model_ptr_, data_ptr_, &opt_, NULL, &cam_, mjCAT_ALL, &scn_);
      for (const auto& a : arrows) {
          mjtNum from[3] = {a.from.x(),  a.from.y(),  a.from.z()};
          mjtNum f[3]    = {a.force.x(), a.force.y(), a.force.z()};
          addForceArrow(from, f, scale, a.rgba);
      }
      mjr_render(viewport, &scn_, &con_);
      glfwSwapBuffers(window_);
      glfwPollEvents();
  }

  void renderWithHandForces(
      const Eigen::Vector3d& p_lhand,
      const Eigen::Vector3d& f_lhand,
      const Eigen::Vector3d& p_rhand,
      const Eigen::Vector3d& f_rhand
  ) {
      // 1. Aggiorna la scena (reset + rebuild geoms da MuJoCo)
      mjrRect viewport = {0, 0, 0, 0};
      glfwGetFramebufferSize(window_, &viewport.width, &viewport.height);
      mjv_updateScene(model_ptr_, data_ptr_, &opt_, NULL, &cam_, mjCAT_ALL, &scn_);

      // 2. Aggiungi frecce mano sinistra (blu) e destra (rossa)
      float rgba_left[4]  = {0.0f, 0.4f, 1.0f, 0.9f};
      float rgba_right[4] = {1.0f, 0.3f, 0.0f, 0.9f};
      float scale = 0.05f; // 0.05 m/N -> a 10 N = freccia 50 cm

      mjtNum from_l[3] = {p_lhand.x(), p_lhand.y(), p_lhand.z()};
      mjtNum f_l[3]    = {f_lhand.x(), f_lhand.y(), f_lhand.z()};
      mjtNum from_r[3] = {p_rhand.x(), p_rhand.y(), p_rhand.z()};
      mjtNum f_r[3]    = {f_rhand.x(), f_rhand.y(), f_rhand.z()};

      addForceArrow(from_l, f_l, scale, rgba_left);
      addForceArrow(from_r, f_r, scale, rgba_right);

      // 3. Render
      mjr_render(viewport, &scn_, &con_);
      glfwSwapBuffers(window_);
      glfwPollEvents();
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