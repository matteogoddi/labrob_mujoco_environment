
// GLFW
#include <GLFW/glfw3.h>
#include <memory>
#include <cstring>

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
    // update scene and render
    mjv_updateScene(model_ptr_, data_ptr_, &opt_, NULL, &cam_, mjCAT_ALL, &scn_);
    finalizeFrame_();
  }

  bool windowShouldClose() {
    return glfwWindowShouldClose(window_);
  }

  // --- Gait control panel: public accessors -------------------------------

  // One-shot event: true exactly once per "Start steps" click, then resets.
  bool consumeStartStepsPressed() {
    bool v = start_steps_pressed_;
    start_steps_pressed_ = false;
    return v;
  }

  bool   isFilterEnabled() const { return filter_enabled_ != 0; }
  void   setFilterEnabled(bool v) { filter_enabled_ = v ? 1 : 0; }

  double getStepLengthX()  const { return step_length_x_; }
  double getStepLengthY()  const { return step_length_y_; }
  double getYawIncrement() const { return yaw_increment_; }
  double getDoubleSupportDuration() const { return double_support_duration_; }
  double getSingleSupportDuration() const { return single_support_duration_; }

  // mouse button callback
  void onMouseButton(GLFWwindow* window, int button, int act, int mods) {
    // update button state
    button_left_   = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)   == GLFW_PRESS);
    button_middle_ = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right_  = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)  == GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx_, &lasty_);

    // route through mjUI: on press, latch which rectangle owns the drag;
    // only forward to mjui_event if the drag started on (or is already
    // owned by) the panel rectangle, mirroring MuJoCo's own simulate.cc.
    updateMjuiInputState_(window);
    uistate_.type = (act == GLFW_PRESS) ? mjEVENT_PRESS : mjEVENT_RELEASE;
    uistate_.button = glfwButtonToMjtButton_(button);
    uistate_.buttontime = glfwGetTime();

    if (act == GLFW_PRESS && uistate_.mouserect != 0) {
      uistate_.dragrect = uistate_.mouserect;
      uistate_.dragbutton = uistate_.button;
    }

    if (uistate_.dragrect == ui0_.rectid ||
        (uistate_.dragrect == 0 && uistate_.mouserect == ui0_.rectid)) {
      mjuiItem* it = mjui_event(&ui0_, &uistate_, &con_);
      if (it != nullptr && it->itemid == kItemStartSteps) {
        start_steps_pressed_ = true;
      }
    }

    if (act == GLFW_RELEASE) {
      uistate_.dragrect = 0;
      uistate_.dragbutton = mjBUTTON_NONE;
    }
  }

  // mouse move callback
  void onMouseMove(GLFWwindow* window, double xpos, double ypos) {
    updateMjuiInputState_(window);
    uistate_.type = mjEVENT_MOVE;

    // a drag that started on the panel stays owned by the panel: forward it
    // and do NOT also move the camera.
    if (uistate_.dragrect == ui0_.rectid) {
      mjui_event(&ui0_, &uistate_, &con_);
      lastx_ = xpos;
      lasty_ = ypos;
      return;
    }

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
    updateMjuiInputState_(window);
    uistate_.type = mjEVENT_SCROLL;
    uistate_.sx = 0;
    uistate_.sy = -yoffset;

    if (uistate_.mouserect == ui0_.rectid) {
      mjui_event(&ui0_, &uistate_, &con_);
      return;
    }

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
      mjv_updateScene(model_ptr_, data_ptr_, &opt_, NULL, &cam_, mjCAT_ALL, &scn_);
      for (const auto& a : arrows) {
          mjtNum from[3] = {a.from.x(),  a.from.y(),  a.from.z()};
          mjtNum f[3]    = {a.force.x(), a.force.y(), a.force.z()};
          addForceArrow(from, f, scale, a.rgba);
      }
      finalizeFrame_();
  }

  void renderWithHandForces(
      const Eigen::Vector3d& p_lhand,
      const Eigen::Vector3d& f_lhand,
      const Eigen::Vector3d& p_rhand,
      const Eigen::Vector3d& f_rhand
  ) {
      // 1. Aggiorna la scena (reset + rebuild geoms da MuJoCo)
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
      finalizeFrame_();
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

    // Contact point/force arrows off by default (too much visual clutter with
    // many self-contacts); toggle on ad hoc when debugging contacts.
    opt_.flags[mjVIS_CONTACTPOINT] = false;
    opt_.flags[mjVIS_CONTACTFORCE] = false;
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

    // --- Gait control panel (mjUI) ---
    std::memset(&ui0_, 0, sizeof(mjUI));
    std::memset(&uistate_, 0, sizeof(mjuiState));
    ui0_.spacing  = mjui_themeSpacing(0);
    ui0_.color    = mjui_themeColor(0);
    ui0_.rectid   = 1;
    ui0_.auxid    = 0;
    ui0_.userdata = this;
    uistate_.userdata = this;

    const mjuiDef defControlPanel[] = {
      {mjITEM_SECTION,   "Gait Control",  1, nullptr,           ""},
      {mjITEM_BUTTON,    "Start steps",   2, nullptr,           ""},
      {mjITEM_SLIDERNUM, "Step len X",    2, &step_length_x_,   "0 0.3"},
      {mjITEM_SLIDERNUM, "Step len Y",    2, &step_length_y_,   "-0.15 0.15"},
      {mjITEM_SLIDERNUM, "Yaw / stride",  2, &yaw_increment_,   "-0.3 0.3"},
      {mjITEM_SLIDERNUM, "DS duration",   2, &double_support_duration_, "400 3000"},
      {mjITEM_SLIDERNUM, "SS duration",   2, &single_support_duration_, "400 3000"},
      {mjITEM_END}
    };
    mjui_add(&ui0_, defControlPanel);

    // --- State estimation panel (mjUI) ---
    // Own section: the EKF toggle isn't a gait parameter, it drives the
    // state estimator that feeds robot_state upstream of gait control /
    // WBC, so it doesn't belong under "Gait Control". Add further
    // estimator-related knobs here (e.g. noise params) the same way.
    const mjuiDef defStateEstimationPanel[] = {
      {mjITEM_SECTION,   "State Estimation", 1, nullptr,          ""},
      {mjITEM_CHECKBYTE, "EKF filter",       2, &filter_enabled_, ""},
      {mjITEM_END}
    };
    mjui_add(&ui0_, defStateEstimationPanel);

    // --- Online planner panel (mjUI) ---
    // Appended as a second collapsible section in the SAME panel (ui0_,
    // rectid 1) -- do NOT re-memset ui0_/uistate_ here, that would wipe out
    // the "Gait Control" section added above. Empty for now: add
    // mjITEM_SLIDERNUM / mjITEM_CHECKBYTE / mjITEM_BUTTON entries here the
    // same way as defControlPanel above, each bound to a new member
    // variable (declared in the private section below, near
    // step_length_x_ etc.) plus a getter, then read that getter from the
    // main loop just like getStepLengthX() is read today.
    const mjuiDef defOnlinePlannerPanel[] = {
      {mjITEM_SECTION,   "Online Planner", 1, nullptr,           ""},
      {mjITEM_END}
    };
    mjui_add(&ui0_, defOnlinePlannerPanel);

    mjui_resize(&ui0_, &con_);
    mjr_addAux(ui0_.auxid, ui0_.width, ui0_.maxheight, ui0_.spacing.samples, &con_);

    updateUiLayout_();
    mjui_update(-1, -1, &ui0_, &uistate_, &con_);
  }

  static std::unique_ptr<MujocoUI> mujoco_ui_ptr_;

 private:
  MujocoUI(const MujocoUI& rhs) = delete;
  MujocoUI& operator=(const MujocoUI& rhs) = delete;

  // Index of the "Start steps" button within the panel's single section
  // (item 0 is the section header itself, so the button is item 1... but
  // mjITEM_SECTION headers are not counted in itemid numbering by mjui_add;
  // itemid is 0-based among actual (non-section) items in the section).
  static constexpr int kItemStartSteps = 0;

  static mjtButton glfwButtonToMjtButton_(int glfw_button) {
    switch (glfw_button) {
      case GLFW_MOUSE_BUTTON_LEFT:   return mjBUTTON_LEFT;
      case GLFW_MOUSE_BUTTON_RIGHT:  return mjBUTTON_RIGHT;
      case GLFW_MOUSE_BUTTON_MIDDLE: return mjBUTTON_MIDDLE;
      default:                       return mjBUTTON_NONE;
    }
  }

  // Keeps uistate_.rect[] (window/panel/viewport) in sync with the current
  // framebuffer size; cheap enough to call once per frame with no dedicated
  // resize callback.
  void updateUiLayout_() {
    int w = 0, h = 0;
    glfwGetFramebufferSize(window_, &w, &h);
    uistate_.nrect = 3;
    uistate_.rect[0] = {0, 0, w, h};
    uistate_.rect[1] = {0, 0, panel_enabled_ ? ui0_.width : 0, h};
    uistate_.rect[2] = {uistate_.rect[1].width, 0, mjMAX(0, w - uistate_.rect[1].width), h};
  }

  // Refreshes the mouse/keyboard-modifier portion of uistate_ from GLFW,
  // mirroring MuJoCo's own PlatformUIAdapter::UpdateMjuiState pattern.
  void updateMjuiInputState_(GLFWwindow* window) {
    uistate_.left   = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)   == GLFW_PRESS);
    uistate_.right  = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)  == GLFW_PRESS);
    uistate_.middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    uistate_.control = (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                        glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS);
    uistate_.shift   = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)   == GLFW_PRESS ||
                        glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)   == GLFW_PRESS);
    uistate_.alt     = (glfwGetKey(window, GLFW_KEY_LEFT_ALT)     == GLFW_PRESS ||
                        glfwGetKey(window, GLFW_KEY_RIGHT_ALT)     == GLFW_PRESS);

    double x = 0.0, y = 0.0;
    glfwGetCursorPos(window, &x, &y);
    int fbw = 0, fbh = 0, ww = 1, wh = 1;
    glfwGetFramebufferSize(window, &fbw, &fbh);
    glfwGetWindowSize(window, &ww, &wh);
    double ratio = ww > 0 ? static_cast<double>(fbw) / ww : 1.0;
    x *= ratio;
    y *= ratio;
    y = uistate_.rect[0].height - y; // flip to OpenGL (origin bottom-left) convention

    updateUiLayout_();

    uistate_.dx = x - uistate_.x;
    uistate_.dy = y - uistate_.y;
    uistate_.x = x;
    uistate_.y = y;
    uistate_.mouserect = mjr_findRect(mju_round(x), mju_round(y),
                                       uistate_.nrect - 1, uistate_.rect + 1) + 1;
  }

  // Shared render tail used by render()/renderWithForces()/renderWithHandForces():
  // shrinks the 3D viewport to leave room for the panel, then draws both.
  void finalizeFrame_() {
    updateUiLayout_();
    mjr_render(uistate_.rect[2], &scn_, &con_);
    if (panel_enabled_) {
      mjui_update(-1, -1, &ui0_, &uistate_, &con_);
      mjui_render(&ui0_, &uistate_, &con_);
    }
    glfwSwapBuffers(window_);
    glfwPollEvents();
  }

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

  // Gait control panel:
  mjUI      ui0_{};
  mjuiState uistate_{};
  bool      panel_enabled_ = true;

  mjtByte filter_enabled_             = 0;
  double  step_length_x_              = 0.1;
  double  step_length_y_              = 0.0;
  double  yaw_increment_              = 0.0;
  double  double_support_duration_    = 2000;
  double  single_support_duration_    = 2000;
  bool    start_steps_pressed_        = false;
};

std::unique_ptr<MujocoUI> MujocoUI::mujoco_ui_ptr_;
} // end namespace labrob