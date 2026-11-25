# Unitree G1 Mode 13 实验指南

MuJoCo 仿真和 G1 真机共用一套 ROS 2 Foxy / RViz2 观察界面。控制器与 RViz2 分别占用一个终端。在同一台 Linux 主机上用同一用户运行，即可在实验中途打开或重启观察端。若同时观察仿真和真机，给两个观察端使用不同的 ROS 域，以免同名话题混合。

## 1. MuJoCo 仿真

**终端 1 · 启动仿真**。从 `build/` 启动，以便程序找到场景和机器人模型。

```bash
source /opt/ros/foxy/setup.bash
cd ~/G1_Experiment/labrob_mujoco_environment/build
./main --sim --rviz
```

如果是新的工作目录或刚修改了控制器，在项目根目录先运行 `cmake -S . -B build` 和 `cmake --build build --target main --parallel 2`。

**终端 2 · 打开 RViz2**。脚本会加载 Foxy。MuJoCo 停在初始画面时，RViz2 就能显示当前姿态，包括 29 个身体关节和 14 个手指关节。要单独重开 RViz2，在此终端按 `Ctrl+C` 后再次运行脚本。

```bash
cd ~/G1_Experiment/labrob_mujoco_environment
./scripts/run_rviz.sh sim
```

**在 MuJoCo 窗口按键**。先点击窗口，再按标题提示逐步按数字键。每个动作完成后，窗口标题和仿真终端都会显示下一步。

| 按键 | 仿真动作 | 在 RViz2 中看什么 |
| --- | --- | --- |
| `1` | 双臂稍向两侧打开，给手指留出空间 | 双肩和手臂姿态同步变化 |
| `2` | 双手握拳 | 两侧 Dex3 手指逐渐弯曲 |
| `3` | 双前臂缓慢抬起，约 6 秒后水平向前 | 可选 `Joint angle` 查看肘关节角度 |
| `4` | 双脚着地，进入关节 PD 站立 | `Joint torque` 开始显示仿真施加的关节力矩 |
| `5` | 切换到全身控制器（WBC）站立 | 出现期望与反馈点、足底力箭头、质心与 ZMP 预测，身体电机开始有 WBC 力矩指令曲线 |
| `6` | 转移重心，向前迈步 | 查看左右脚规划落点，以及换重心时的横向 ZMP 跟踪 |

## 2. G1 真机实验

**终端 1 · 连接并观察真机**。先连接 G1 的实验网口，用 `ip -br addr` 查看网卡名，确认 `~/unitree_ros2/setup.sh` 中的 `NetworkInterface` 与之相符。这个终端只读 Unitree 状态和总线上的电机指令。

```bash
cd ~/G1_Experiment/labrob_mujoco_environment
source ~/unitree_ros2/setup.sh
./scripts/run_rviz.sh robot
```

收到 `/lowstate` 后，确认状态提示中有 `G1 mode 13 | encoders live`。当前模型对应 `mode_machine=13`、串联关节模式 `mode_pr=0`。RViz2 显示 29 个身体关节，手部显示到手掌。仅检查关节姿态、IMU 或已有的 `/lowcmd` 时，到这一步即可。

**终端 2 · 运行真机 WBC 实验**。在原生 Unitree SDK2 控制终端运行下列命令，并输入与观察端相同的网卡名。**这个程序会检查机型、释放已有运动服务，并向真机发送低层电机指令。**

```bash
cd ~/G1_Experiment/labrob_mujoco_environment/build
read -r -p "连接 G1 的网卡名： " g1_iface
./main --robot "$g1_iface" --rviz
```

启动时按本次实验配置选择初始闭环选项，可用空格分隔：`1` 为质心闭环，`2` 为全身闭环，`3` 为扩展卡尔曼滤波器（EKF）。回车确认后按手柄操作。RViz2 中的 `WBC live` 表示已收到与编码器时间对齐的控制器数据。

| 手柄键 | 程序行为 |
| --- | --- |
| `A` | 首次按下启动 IMU 标定。后续在质心闭环和 EKF 都已启用时，切换全身闭环。 |
| `X` | EKF 已启用时，切换质心闭环。 |
| `B` | 请求切换步行状态。站立时请求开始迈步。 |
| `Y` | 进入程序退出流程，控制终端会询问是否保存日志。 |

**选择观察坐标系**。默认的 `pelvis` 适合检查关节和足端相对骨盆的位置。要看整机在空间中走了多远，需有基座定位。以下参数可追加在 `./scripts/run_rviz.sh robot` 后。

| 定位来源 | RViz2 用法 |
| --- | --- |
| 编码器和 IMU | 保持默认 `pelvis` 看相对姿态。使用 `fixed_frame:=g1_attitude` 可在随骨盆平移的水平坐标系中看机身倾斜。 |
| 控制器 EKF | 默认 `base_source:=controller`。有有效的 `world → pelvis` 后，将 RViz2 的 `Fixed Frame` 设为 `world`。 |
| 外部定位或动捕 | 使用 `base_source:=odometry fixed_frame:=world`，向 `/g1_observer/base_odometry` 提供已配准到控制器世界系的 `world → pelvis` 里程计。 |
| Unitree 运动里程计 | 使用 `base_source:=sport fixed_frame:=unitree_odom`。要比较世界坐标下的跟踪误差，还需标定 `world → unitree_odom`。 |

## 3. RViz2 使用说明与图例

### 3D 视图：箭头、圆点和轨迹

在左侧 `Displays → Walking diagnostics → Namespaces` 中可单独开关每类标记。**力箭头**的起点表示作用位置，方向表示力的方向，长度比例为 **1 mm/N**，例如 100 N 对应 10 cm。

| 标记 | RViz 名称 | 表示什么 |
| --- | --- | --- |
| 粉色箭头 | `contact` | MuJoCo 每个接触点的接触力。一只脚踩地时可能出现多支箭头。 |
| 青蓝色箭头 | `wbc` | WBC 为左、右脚分别求解的支撑合力。 |
| 紫色箭头 | `estimated` | 残差估计器得到的每只脚的支撑合力。与同一只脚的青蓝色箭头比较，可检查估计值与 WBC 解的差异。 |
| 红色箭头 | `external` | 施加到 MuJoCo 机器人刚体上的外力，例如推躯干时的扰动力。 |
| 半透明橙色大圆点 | `desired_com`、`desired_left_foot`、`desired_right_foot` | 当前控制周期内，质心和左右脚底的期望位置。 |
| 青色小圆点 | `feedback_com`、`feedback_left_foot`、`feedback_right_foot` | 对应部位的反馈位置。在同一坐标系下，青点与橙点之间的距离表示该部位的位置跟踪误差大小。 |
| 绿色圆点和绿线 | `current_com`、`mpc_com` | 绿点是模型预测控制器（MPC）的当前滤波质心位置，绿线是从该点开始的预测轨迹。 |
| 黄色圆点和黄线 | `current_zmp`、`mpc_zmp` | 黄点是 MPC 的当前滤波零力矩点（ZMP），黄线是未来的 ZMP 预测轨迹。 |
| 蓝色、橙色矩形框 | `planned_left`、`planned_right` | 未来的落脚位置和朝向。蓝框为左脚，橙框为右脚。 |

真机收到有效位置指令时，还会用**橙色骨架**显示已启用位置控制的指令关节链。若一条腿的各关节都启用了位置控制，**紫色小方块**显示由 `LowCmd` 算出的骨盆相对足端目标。检查坐标系时，打开 `Displays → Compare TF frames` 可对照机器人模型坐标系与控制器的 `wbc_*` 坐标系。

### 右侧 Live plots：看过去 10 秒的跟踪效果

默认上图是横向 ZMP，下图是腰部偏航关节力矩。上下两张图可以独立选择信号。先选曲线类型，再为关节曲线选电机名。横轴右端 `0` 表示现在，图下方显示最新值，`—` 表示暂时没有新样本。切换信号后，新曲线从切换时刻开始记录。

| 曲线 | 橙色虚线 | 青色实线 |
| --- | --- | --- |
| `ZMP X` / `ZMP Y`（m） | `Ref` 为控制器本周期的 ZMP 参考 | `Contact` 为仿真左右足接触合力与合力矩在参考水平面上算出的 ZMP，法向合力大于 1 N 时发布。真机目前只提供同步 WBC 的参考线。 |
| `Joint torque`（N·m） | 仿真为 WBC 力矩指令，真机为 `LowCmd.tau` 前馈力矩 | 仿真为已施加的关节驱动力矩 `qfrc_actuator`，真机为 `LowState.tau_est` 估计力矩。 |
| `Joint angle` / `Joint velocity`（rad / rad/s） | 真机有同步 `LowCmd` 时显示指令角度或速度 | 来自关节状态的角度或速度，从仿真初始姿态或有效真机编码器数据开始显示。 |

**看左右换重心**，上图选 `ZMP Y` 比较参考与接触反馈，配合 3D 视图里的黄色预测轨迹看下一步趋势。**查腰部抖动**，两张图选同一个腰部关节，例如 `waist_pitch_joint`，分别画 `Joint angle` 和 `Joint torque`，即可对照角度和力矩变化的时刻。
