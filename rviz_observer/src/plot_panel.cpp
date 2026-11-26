#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <memory>
#include <vector>

#include <QComboBox>
#include <QLabel>
#include <QSignalBlocker>
#include <QTimer>
#include <QVBoxLayout>
#include <qwt_plot.h>
#include <qwt_plot_curve.h>
#include <qwt_plot_grid.h>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>
#include <rviz_common/display_context.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

namespace g1_wbc_observer {
using JointState = sensor_msgs::msg::JointState;
using PointStamped = geometry_msgs::msg::PointStamped;
constexpr double history = 10.0;
enum Signal { ZmpX, ZmpY, Torque, Angle, Speed };

class Chart : public QWidget {
 public:
  explicit Chart(Signal initial, QWidget* parent) : QWidget(parent) {
    auto* column = new QVBoxLayout(this);
    column->setContentsMargins(2, 2, 2, 2);
    signal = new QComboBox(this);
    QFont heading = font();
    heading.setPixelSize(20);
    heading.setBold(true);
    signal->setFont(heading);
    signal->addItems({"ZMP X", "ZMP Y", "Joint torque", "Joint angle", "Joint velocity"});
    signal->setCurrentIndex(initial);
    joint = new QComboBox(this);
    joint->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
    joint->setMinimumContentsLength(8);
    column->addWidget(signal);
    column->addWidget(joint);

    plot = new QwtPlot(this);
    plot->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    plot->setMinimumSize(145, 85);
    plot->setCanvasBackground(QColor("#202731"));
    plot->setAxisScale(QwtPlot::xBottom, -history, 0, 2);
    plot->setAxisMaxMajor(QwtPlot::yLeft, 5);
    plot->setAxisFont(QwtPlot::xBottom, font());
    plot->setAxisFont(QwtPlot::yLeft, font());
    auto* grid = new QwtPlotGrid;
    grid->setMajorPen(QPen(QColor("#596471"), 0, Qt::DotLine));
    grid->attach(plot);
    for (int i = 0; i < 2; ++i) {
      curves[i] = new QwtPlotCurve;
      // The dashed command stays distinguishable when both signals overlap.
      curves[i]->setPen(QPen(i == 0 ? QColor("#ffab52") : QColor("#5cd5ed"),
                             i == 0 ? 2.1 : 1.8, i == 0 ? Qt::DashLine : Qt::SolidLine));
      curves[i]->setZ(i == 0 ? 2 : 1);
      curves[i]->setRenderHint(QwtPlotItem::RenderAntialiased);
      curves[i]->attach(plot);
    }
    column->addWidget(plot, 1);
    values = new QLabel(this);
    QFont legend_font = font();
    legend_font.setPixelSize(22);
    values->setFont(legend_font);
    values->setWordWrap(true);
    column->addWidget(values);
    connect(signal, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this] { reset(); });
    connect(joint, &QComboBox::currentTextChanged, this, [this] { reset(); });
    reset();
  }

  void discover(const std::vector<std::string>& names) {
    if (names == joint_names) return;
    joint_names = names;
    QString selected = joint->currentText();
    if (selected.isEmpty()) selected = "waist_yaw_joint";
    QStringList choices;
    for (const auto& name : names) choices << QString::fromStdString(name);
    choices.sort();
    QSignalBlocker block(joint);
    joint->clear();
    joint->addItems(choices);
    if (choices.contains(selected)) joint->setCurrentText(selected);
    else if (isJoint()) reset();
  }

  void addJoint(const JointState& msg, bool command) {
    if (!isJoint()) return;
    const auto& data = signal->currentIndex() == Torque ? msg.effort :
                       signal->currentIndex() == Angle ? msg.position : msg.velocity;
    auto it = std::find(msg.name.begin(), msg.name.end(), joint->currentText().toStdString());
    const size_t index = std::distance(msg.name.begin(), it);
    if (it != msg.name.end() && index < data.size())
      append(command ? 0 : 1, msg.header.stamp, data[index]);
  }

  void addZmp(const PointStamped& msg, bool desired) {
    if (!isJoint() && msg.header.frame_id == "world")
      append(desired ? 0 : 1, msg.header.stamp, signal->currentIndex() == ZmpX ? msg.point.x : msg.point.y);
  }

  void refresh(double now) {
    QStringList labels;
    for (int i = 0; i < 2; ++i) {
      auto& series = samples[i];
      while (!series.empty() && series.front().x() < now - history) series.pop_front();
      QVector<QPointF> visible;
      visible.reserve(static_cast<int>(series.size()));
      for (const auto& p : series) visible.append(QPointF(p.x() - now, p.y()));
      curves[i]->setSamples(visible);
      const QString name = isJoint() ? (i == 0 ? "Cmd" : "Feedback") : (i == 0 ? "Ref" : "Contact");
      const QString val = !series.empty() && now - series.back().x() < 0.5 ?
          QString::number(series.back().y(), 'f', 3) : QStringLiteral("—");
      labels << QString("<b style='color:%1'>%2</b>&nbsp;%3").arg(i == 0 ? "#955000" : "#00697b", name, val);
    }
    values->setText(labels.join(" &nbsp; "));
    plot->replot();
  }

 private:
  bool isJoint() const { return signal->currentIndex() >= Torque; }

  void reset() {
    for (auto& series : samples) series.clear();
    joint->setVisible(isJoint());
    const QStringList units = {"m", "m", "N·m", "rad", "rad/s"};
    QwtText title(units[signal->currentIndex()]);
    title.setFont(font());
    plot->setAxisTitle(QwtPlot::yLeft, title);
  }

  void append(int channel, const builtin_interfaces::msg::Time& stamp, double value) {
    if (!std::isfinite(value)) return;
    const double t = rclcpp::Time(stamp).seconds();
    auto& series = samples[channel];
    if (!series.empty() && (t < series.back().x() || t - series.back().x() > 0.5))
      series.clear();
    if (!series.empty() && t == series.back().x()) series.pop_back();
    series.emplace_back(t, value);
    while (series.size() > 4096 || t - series.front().x() > history) series.pop_front();
  }

  QComboBox *signal, *joint;
  QLabel* values;
  QwtPlot* plot;
  std::array<QwtPlotCurve*, 2> curves;
  std::array<std::deque<QPointF>, 2> samples;
  std::vector<std::string> joint_names;
};

class PlotPanel : public rviz_common::Panel {
 public:
  PlotPanel() {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(4, 4, 4, 4);
    QFont panel_font = font();
    panel_font.setPixelSize(18);
    setFont(panel_font);
    for (auto initial : {ZmpY, Torque}) {
      auto* chart = new Chart(initial, this);
      charts.push_back(chart);
      layout->addWidget(chart, 1);
    }
    auto* caption = new QLabel("Time (s) relative to now", this);
    caption->setAlignment(Qt::AlignCenter);
    layout->addWidget(caption);
  }

  void onInitialize() override {
    // Service the plot subscriptions on the Qt thread. Only this timer accesses the chart buffers.
    rclcpp::NodeOptions options;
    options.use_global_arguments(false);
    node = std::make_shared<rclcpp::Node>("live_plots", "/g1_observer", options);
    executor.add_node(node);
    const auto qos = rclcpp::SensorDataQoS().keep_last(16);
    for (const bool command : {false, true}) {
      subscriptions.push_back(node->create_subscription<JointState>(
          command ? "command/joint_states" : "joint_states", qos,
          [this, command](JointState::ConstSharedPtr msg) {
            for (auto* chart : charts) {
              if (!command) chart->discover(msg->name);
              chart->addJoint(*msg, command);
            }
          }));
    }
    for (const bool desired : {false, true}) {
      subscriptions.push_back(node->create_subscription<PointStamped>(
          desired ? "desired/zmp" : "feedback/contact_zmp", qos,
          [this, desired](PointStamped::ConstSharedPtr msg) {
            for (auto* chart : charts) chart->addZmp(*msg, desired);
          }));
    }
    auto* timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, [this] {
      executor.spin_some(std::chrono::milliseconds(5));
      for (auto* chart : charts) chart->refresh(node->now().seconds());
    });
    timer->start(50);
    if (qEnvironmentVariableIntValue("G1_REPORT_FPS") == 1) {
      using Clock = std::chrono::steady_clock;
      auto* monitor = new QTimer(this);
      connect(monitor, &QTimer::timeout, this,
              [this, previous = getDisplayContext()->getFrameCount(), start = Clock::now()]() mutable {
        const auto now = Clock::now();
        const auto frames = getDisplayContext()->getFrameCount();
        const double seconds = std::chrono::duration<double>(now - start).count();
        RCLCPP_INFO(node->get_logger(), "RViz rendering: %.1f FPS", (frames - previous) / seconds);
        previous = frames;
        start = now;
      });
      monitor->start(5000);
    }
  }

 private:
  std::vector<Chart*> charts;
  rclcpp::Node::SharedPtr node;
  rclcpp::executors::SingleThreadedExecutor executor;
  std::vector<rclcpp::SubscriptionBase::SharedPtr> subscriptions;
};
}  // namespace g1_wbc_observer

PLUGINLIB_EXPORT_CLASS(g1_wbc_observer::PlotPanel, rviz_common::Panel)
