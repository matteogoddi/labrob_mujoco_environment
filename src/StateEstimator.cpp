#include <StateEstimator.hpp>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <utils.hpp>

namespace labrob {

StateEstimator::StateEstimator(const pinocchio::Model& model,
                               double dt,
                               Filter filter,
                               const NoiseParams& noise)
    : model_(model), dt_(dt), filter_(filter)
{
    njnt_ = model_.nv - 6;

    Eigen::VectorXd q0 = pinocchio::neutral(model_);

    if (filter_ == Filter::BaseEKF) {
        base_ekf_ = std::make_unique<BaseEKF>(model_, q0, dt_);
    } else if (filter_ == Filter::SimpleEKF) {
        simple_ekf_ = std::make_unique<SimpleEKF>(model_, q0, dt_);
    } else {
        joint_kf_ = std::make_unique<JointKF>(dt_, njnt_);
        if (filter_ == Filter::RightInvariantEKF) {
            std::array<RightInvariantEKF::FootConfig, 2> feet = {{
                {"left_foot_link",  0},
                {"right_foot_link", 1}
            }};
            ri_ekf_ = std::make_unique<RightInvariantEKF>(model_, q0, dt_, feet, noise);
        } else {
            std::array<DiligentKio::FootConfig, 2> feet = {{
                {"left_foot_link",  0},
                {"right_foot_link", 1}
            }};
            diligent_kio_ = std::make_unique<DiligentKio>(model_, q0, dt_, feet, noise);
        }
    }
}

void StateEstimator::activate(const RobotState& robot_state, const Eigen::VectorXd& q_joints)
{
    if (active_) return;
    active_ = true;

    if (filter_ == Filter::BaseEKF) {
        // Build full Pinocchio config: identity orientation, position from odometry.
        Eigen::VectorXd q_init = pinocchio::neutral(model_);
        q_init.head<3>()   = robot_state.position;
        q_init.tail(njnt_) = q_joints;

        pinocchio::Data data(model_);
        pinocchio::forwardKinematics(model_, data, q_init);
        pinocchio::framesForwardKinematics(model_, data, q_init);

        const Eigen::Vector3d pL =
            data.oMf[model_.getFrameId("left_foot_link")].translation();
        const Eigen::Vector3d pR =
            data.oMf[model_.getFrameId("right_foot_link")].translation();

        base_ekf_->initialize(q_init, pL, pR);
    } else if (filter_ == Filter::RightInvariantEKF) {
        ri_ekf_->addContact(0, q_joints);
        ri_ekf_->addContact(1, q_joints);
    } else if (filter_ == Filter::DiligentKio) {
        diligent_kio_->addContact(0, q_joints);
        diligent_kio_->addContact(1, q_joints);
    }
    // SimpleEKF: no initialisation needed
}

void StateEstimator::update(RobotState& robot_state,
                            const Eigen::Vector3d& gyro,
                            const Eigen::Vector3d& acc,
                            const std::array<bool,2>& contact,
                            const Eigen::VectorXd& wbc_q_ddot)
{
    if (!active_) return;

    if (filter_ == Filter::BaseEKF) {
        Eigen::VectorXd jnt_pos(njnt_), jnt_vel(njnt_);
        for (int i = 0; i < njnt_; ++i) {
            const std::string& name = model_.names[i + 2];
            jnt_pos(i) = robot_state.joint_state.at(name).pos;
            jnt_vel(i) = robot_state.joint_state.at(name).vel;
        }

        base_ekf_->filter(acc, gyro, jnt_pos, jnt_vel, wbc_q_ddot,
                          contact[0], contact[1]);

        robot_state.position         = base_ekf_->getBasePosition();
        robot_state.linear_velocity  = base_ekf_->getBaseVelocity();
        robot_state.orientation      = base_ekf_->getBaseOrientation();
        robot_state.angular_velocity = base_ekf_->getBaseOmega();
        return;
    }

    if (filter_ == Filter::SimpleEKF) {
        // Build observation: [base_pos(3), imu_orientation_rotvec(3), joint_pos(njnt_)]
        Eigen::VectorXd y_actual(6 + njnt_);
        y_actual.head(3) = robot_state.position;
        y_actual.segment(3, 3) = rotVecFromQuaternion(robot_state.orientation);
        for (int i = 0; i < njnt_; ++i) {
            const std::string& name = model_.names[i + 2];
            y_actual(6 + i) = robot_state.joint_state.at(name).pos;
        }

        simple_ekf_->filter(y_actual, wbc_q_ddot);
        RobotState est = simple_ekf_->getState();

        robot_state.position         = est.position;
        robot_state.orientation      = est.orientation;
        robot_state.linear_velocity  = est.linear_velocity;
        robot_state.angular_velocity = est.angular_velocity;
        for (int i = 0; i < njnt_; ++i) {
            const std::string& name = model_.names[i + 2];
            robot_state.joint_state.at(name).pos = est.joint_state.at(name).pos;
            robot_state.joint_state.at(name).vel = est.joint_state.at(name).vel;
        }
        return;
    }

    // JointKF — pre-filters joint positions/velocities (used by RI-EKF and DiligentKio)
    {
        Eigen::VectorXd jnt_pos(njnt_);
        Eigen::VectorXd jnt_vel(njnt_);
        for (int i = 0; i < njnt_; ++i) {
            const std::string& name = model_.names[i + 2];
            jnt_pos(i) = robot_state.joint_state.at(name).pos;
            jnt_vel(i) = robot_state.joint_state.at(name).vel;
        }

        Eigen::VectorXd input1(njnt_ + 3);
        input1 << jnt_pos, gyro;
        Eigen::VectorXd input2(njnt_ + 3);
        input2 << wbc_q_ddot.tail(njnt_), wbc_q_ddot.segment(3, 3);
        joint_kf_->filter(input1, input2);
    }

    const Eigen::VectorXd& filt_pos = joint_kf_->getFilteredJointPositions();
    const Eigen::VectorXd& filt_vel = joint_kf_->getFilteredJointVelocities();

    if (filter_ == Filter::RightInvariantEKF) {
        ri_ekf_->filter(gyro, acc, filt_pos, filt_vel, contact);
        robot_state.position         = ri_ekf_->getPosition();
        robot_state.orientation      = ri_ekf_->getQuaternion();
        robot_state.linear_velocity  = ri_ekf_->getVelocity();
        robot_state.angular_velocity = ri_ekf_->getOmegaBody();
    } else {
        diligent_kio_->filter(gyro, acc, filt_pos, filt_vel, contact);
        robot_state.position         = diligent_kio_->getPosition();
        robot_state.orientation      = diligent_kio_->getQuaternion();
        robot_state.linear_velocity  = diligent_kio_->getVelocity();
        robot_state.angular_velocity = diligent_kio_->getOmegaBody();
    }

    for (int i = 0; i < njnt_; ++i) {
        const std::string& name = model_.names[i + 2];
        robot_state.joint_state.at(name).pos = filt_pos(i);
        robot_state.joint_state.at(name).vel = filt_vel(i);
    }
}

} // namespace labrob