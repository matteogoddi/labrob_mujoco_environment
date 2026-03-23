#include <hrp4_locomotion/EstimateForce.hpp>

#include <array>

#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/rnea.hpp>

#include <hrp4_locomotion/utils.hpp>

namespace {

std::vector<int>
buildVelocityIndices(
        const pinocchio::Model& model,
        const std::array<const char*, 7>& joint_names
) {
    std::vector<int> velocity_indices;
    velocity_indices.reserve(joint_names.size());

    for (const auto* joint_name : joint_names) {
        if (!model.existJointName(joint_name)) {
            continue;
        }

        const auto joint_id = model.getJointId(joint_name);
        const auto& joint = model.joints[joint_id];
        if (joint.nv() == 1) {
            velocity_indices.push_back(joint.idx_v());
        }
    }

    return velocity_indices;
}

} // end anonymous namespace

namespace labrob {

EstimateForce::EstimateForce()
        : initialized_(false),
            robot_model_(),
            robot_data_(robot_model_),
            left_wrist_frame_id_(0),
            right_wrist_frame_id_(0),
            left_wrench_(Eigen::VectorXd::Zero(6)),
            right_wrench_(Eigen::VectorXd::Zero(6)),
            left_wrench_filtered_(Eigen::VectorXd::Zero(6)),
            right_wrench_filtered_(Eigen::VectorXd::Zero(6)),
            left_wrench_bias_(Eigen::VectorXd::Zero(6)),
            right_wrench_bias_(Eigen::VectorXd::Zero(6)),
            bias_sample_count_(0),
            bias_sample_target_(500),
            alpha_(0.1),
            damping_(1e-4) {
}

EstimateForce::EstimateForce(const pinocchio::Model& robot_model)
        : EstimateForce() {
    initialize(robot_model);
}

void
EstimateForce::initialize(const pinocchio::Model& robot_model) {
    robot_model_ = robot_model;
    robot_data_ = pinocchio::Data(robot_model_);

    static const std::array<const char*, 7> left_arm_joint_names = {
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint"
    };
    static const std::array<const char*, 7> right_arm_joint_names = {
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint"
    };

    left_arm_velocity_indices_ = buildVelocityIndices(robot_model_, left_arm_joint_names);
    right_arm_velocity_indices_ = buildVelocityIndices(robot_model_, right_arm_joint_names);

    left_wrist_frame_id_ = robot_model_.existFrame("left_wrist_yaw_joint") ?
            robot_model_.getFrameId("left_wrist_yaw_joint") : 0;
    right_wrist_frame_id_ = robot_model_.existFrame("right_wrist_yaw_joint") ?
            robot_model_.getFrameId("right_wrist_yaw_joint") : 0;

    tau_model_ = Eigen::VectorXd::Zero(robot_model_.nv);
    tau_residual_ = Eigen::VectorXd::Zero(robot_model_.nv);
    left_tau_res_ = Eigen::VectorXd::Zero(left_arm_velocity_indices_.size());
    right_tau_res_ = Eigen::VectorXd::Zero(right_arm_velocity_indices_.size());

    left_wrench_.setZero(6);
    right_wrench_.setZero(6);
    left_wrench_filtered_.setZero(6);
    right_wrench_filtered_.setZero(6);
    left_wrench_bias_.setZero(6);
    right_wrench_bias_.setZero(6);
    bias_sample_count_ = 0;

    initialized_ = true;
}

void
EstimateForce::computeModelTorque(
        const Eigen::VectorXd& q,
        const Eigen::VectorXd& dq,
        const Eigen::VectorXd& ddq
) {
    tau_model_ = pinocchio::rnea(robot_model_, robot_data_, q, dq, ddq);
}

void
EstimateForce::computeResidualTorque(const Eigen::VectorXd& tau_measured) {
    tau_residual_ = tau_measured - tau_model_;
}

Eigen::VectorXd
EstimateForce::selectArmResiduals(const std::vector<int>& arm_velocity_indices) const {
    Eigen::VectorXd tau_res_arm = Eigen::VectorXd::Zero(arm_velocity_indices.size());
    for (std::size_t i = 0; i < arm_velocity_indices.size(); ++i) {
        const int index = arm_velocity_indices[i];
        if (index >= 0 && index < tau_residual_.size()) {
            tau_res_arm(static_cast<Eigen::Index>(i)) = tau_residual_(index);
        }
    }
    return tau_res_arm;
}

Eigen::VectorXd
EstimateForce::computeEstimatedWrench(
        const Eigen::MatrixXd& jacobian,
        const Eigen::VectorXd& tau_res
) const {
    if (tau_res.size() == 0 || jacobian.cols() == 0) {
        return Eigen::VectorXd::Zero(6);
    }

    const Eigen::MatrixXd A = jacobian * jacobian.transpose() +
                                                        damping_ * Eigen::MatrixXd::Identity(6, 6);
    return A.ldlt().solve(jacobian * tau_res);
}

void
EstimateForce::update(const labrob::RobotState& robot_state) {
    if (!initialized_) {
        return;
    }

    const Eigen::VectorXd q = robot_state_to_pinocchio_joint_configuration(robot_model_, robot_state);
    const Eigen::VectorXd dq = robot_state_to_pinocchio_joint_velocity(robot_model_, robot_state);

    Eigen::VectorXd ddq = Eigen::VectorXd::Zero(robot_model_.nv);
    Eigen::VectorXd tau_measured = Eigen::VectorXd::Zero(robot_model_.nv);
    for (pinocchio::JointIndex joint_id = 2;
             joint_id < static_cast<pinocchio::JointIndex>(robot_model_.njoints);
             ++joint_id) {
        const auto& joint_name = robot_model_.names[joint_id];
        const int velocity_index = robot_model_.joints[joint_id].idx_v();
        ddq(velocity_index) = robot_state.joint_state[joint_name].acc;
        tau_measured(velocity_index) = robot_state.joint_state[joint_name].eff;
    }

    pinocchio::computeJointJacobians(robot_model_, robot_data_, q);
    pinocchio::framesForwardKinematics(robot_model_, robot_data_, q);

    Eigen::MatrixXd J_left = Eigen::MatrixXd::Zero(6, robot_model_.nv);
    Eigen::MatrixXd J_right = Eigen::MatrixXd::Zero(6, robot_model_.nv);
    pinocchio::getFrameJacobian(
            robot_model_,
            robot_data_,
            left_wrist_frame_id_,
            pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
            J_left
    );
    pinocchio::getFrameJacobian(
            robot_model_,
            robot_data_,
            right_wrist_frame_id_,
            pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
            J_right
    );


    computeModelTorque(q, dq, ddq);
    computeResidualTorque(tau_measured);

    left_tau_res_ = selectArmResiduals(left_arm_velocity_indices_);
    right_tau_res_ = selectArmResiduals(right_arm_velocity_indices_);

    Eigen::MatrixXd J_left_arm(6, left_arm_velocity_indices_.size());
    for (std::size_t col = 0; col < left_arm_velocity_indices_.size(); ++col) {
        J_left_arm.col(static_cast<Eigen::Index>(col)) = J_left.col(left_arm_velocity_indices_[col]);
    }

    Eigen::MatrixXd J_right_arm(6, right_arm_velocity_indices_.size());
    for (std::size_t col = 0; col < right_arm_velocity_indices_.size(); ++col) {
        J_right_arm.col(static_cast<Eigen::Index>(col)) = J_right.col(right_arm_velocity_indices_[col]);
    }

    const Eigen::VectorXd left_wrench_raw = computeEstimatedWrench(J_left_arm, left_tau_res_);
    const Eigen::VectorXd right_wrench_raw = computeEstimatedWrench(J_right_arm, right_tau_res_);

    if (bias_sample_count_ < bias_sample_target_) {
        ++bias_sample_count_;
        const double gain = 1.0 / static_cast<double>(bias_sample_count_);
        left_wrench_bias_ += gain * (left_wrench_raw - left_wrench_bias_);
        right_wrench_bias_ += gain * (right_wrench_raw - right_wrench_bias_);
    }

    left_wrench_ = left_wrench_raw - left_wrench_bias_;
    right_wrench_ = right_wrench_raw - right_wrench_bias_;

    left_wrench_filtered_ = alpha_ * left_wrench_ + (1.0 - alpha_) * left_wrench_filtered_;
    right_wrench_filtered_ = alpha_ * right_wrench_ + (1.0 - alpha_) * right_wrench_filtered_;
}

const Eigen::VectorXd&
EstimateForce::getLeftWristWrench() const {
    return left_wrench_;
}

const Eigen::VectorXd&
EstimateForce::getRightWristWrench() const {
    return right_wrench_;
}

const Eigen::VectorXd&
EstimateForce::getLeftWristWrenchFiltered() const {
    return left_wrench_filtered_;
}

const Eigen::VectorXd&
EstimateForce::getRightWristWrenchFiltered() const {
    return right_wrench_filtered_;
}

} // end namespace labrob
