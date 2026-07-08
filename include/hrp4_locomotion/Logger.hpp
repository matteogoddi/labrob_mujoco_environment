#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstddef>
#include <filesystem>
#include <sstream>
#include <iomanip>

#include <Eigen/Dense>

namespace labrob {

struct WBCEntry {
    double time_ms = 0.0;

    double com_x = 0.0;
    double com_y = 0.0;
    double com_z = 0.0;

    double com_x_des = 0.0;
    double com_y_des = 0.0;
    double com_z_des = 0.0;

    double com_vel_x = 0.0;
    double com_vel_y = 0.0;
    double com_vel_z = 0.0;

    double com_vel_x_des = 0.0;
    double com_vel_y_des = 0.0;
    double com_vel_z_des = 0.0;

    double l_sole_x = 0.0;
    double l_sole_y = 0.0;
    double l_sole_z = 0.0;

    double l_sole_x_des = 0.0;
    double l_sole_y_des = 0.0;
    double l_sole_z_des = 0.0;

    double r_sole_x = 0.0;
    double r_sole_y = 0.0;
    double r_sole_z = 0.0;

    double r_sole_x_des = 0.0;
    double r_sole_y_des = 0.0;
    double r_sole_z_des = 0.0;

    double q_w_pelvis = 0.0;
    double q_x_pelvis = 0.0;
    double q_y_pelvis = 0.0;
    double q_z_pelvis = 0.0;
    
    double q_w_torso = 0.0;
    double q_x_torso = 0.0;
    double q_y_torso = 0.0;
    double q_z_torso = 0.0;
};


struct TimingEntry {
    long time_wbc_us = 0;
    long total_time_us = 0;
};

class Logger {
public:
    Logger() = default;

    explicit Logger(std::size_t reserve_size) {
        reserve(reserve_size);
    }

    void reserve(std::size_t n) {
        wbc_entries_.reserve(n);
        timing_entries_.reserve(n);
    }

    void clear() {
        wbc_entries_.clear();
        timing_entries_.clear();
    }

    void log_wbc_data(const WBCEntry& entry) {
        wbc_entries_.emplace_back(entry);
    }

    void log_wbc_data(WBCEntry&& entry) {
        wbc_entries_.emplace_back(std::move(entry));
    }

    void log_timing_data(const TimingEntry& entry) {
        timing_entries_.emplace_back(entry);
    }

    void log_timing_data(TimingEntry&& entry) {
        timing_entries_.emplace_back(std::move(entry));
    }

    void save_log_data(const std::string& directory = "/tmp") const {
        namespace fs = std::filesystem;
        fs::create_directories(directory);

        auto start_wbc_log = std::chrono::system_clock::now();
        
        save_wbc_logs(directory + "/wbc_log.txt");
        auto end_wbc_log = std::chrono::system_clock::now();
        auto wbc_log_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_wbc_log - start_wbc_log).count();
        std::cout << "WBC logs saved in " << wbc_log_time << " ms" << std::endl;
        
        save_timing_logs(directory + "/timing_log.txt");
        auto end_timing_log = std::chrono::system_clock::now();
        auto timing_log_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_timing_log - end_wbc_log).count();
        std::cout << "timing logs saved in " << timing_log_time << " ms" << std::endl;
    }

    std::size_t wbc_size() const { return wbc_entries_.size(); }
    std::size_t timing_size() const { return timing_entries_.size(); }

private:

    void save_wbc_logs(const std::string& filename) const {
        std::ofstream out(filename);
        if (!out.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        out << "time_ms,"
            << "com_x,com_y,com_z,"
            << "com_x_des,com_y_des,com_z_des,"
            << "com_vel_x,com_vel_y,com_vel_z,"
            << "com_vel_x_des,com_vel_y_des,com_vel_z_des,"

            << "l_sole_x_x,l_sole_x_y,l_sole_x_z,"
            << "l_sole_x_x_des,l_sole_x_y_des,l_sole_x_z_des,"
            << "r_sole_x_x,r_sole_x_y,r_sole_x_z,"
            << "r_sole_x_x_des,r_sole_x_y_des,r_sole_x_z_des,"

            << "q_w_pelvis,q_x_pelvis,q_y_pelvis,q_z_pelvis,"
            << "q_w_torso,q_x_torso,q_y_torso,q_z_torso"
            << "\n";

        for (const auto& e : wbc_entries_) {
            out << e.time_ms << ","
                << e.com_x << "," << e.com_y << "," << e.com_z << ","
                << e.com_x_des << "," << e.com_y_des << "," << e.com_z_des << ","
                << e.com_vel_x << "," << e.com_vel_y << "," << e.com_vel_z << ","
                << e.com_vel_x_des << "," << e.com_vel_y_des << "," << e.com_vel_z_des << ","
                << e.l_sole_x << "," << e.l_sole_y << "," << e.l_sole_z << ","
                << e.l_sole_x_des << "," << e.l_sole_y_des << "," << e.l_sole_z_des << ","
                << e.r_sole_x << "," << e.r_sole_y << "," << e.r_sole_z << ","
                << e.r_sole_x_des << "," << e.r_sole_y_des << "," << e.r_sole_z_des << ","
                << e.q_w_pelvis << "," << e.q_x_pelvis << "," << e.q_y_pelvis << "," << e.q_z_pelvis << ","
                << e.q_w_torso << "," << e.q_x_torso << "," << e.q_y_torso << "," << e.q_z_torso
                << "\n";
        }
    }


    void save_timing_logs(const std::string& filename) const {
        std::ofstream out(filename);
        if (!out.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        out << "time_wbc_us,total_time_us\n";

        for (const auto& e : timing_entries_) {
            out << e.time_wbc_us << ","
                << e.total_time_us
                << "\n";
        }
    }

private:
    std::vector<WBCEntry> wbc_entries_;
    std::vector<TimingEntry> timing_entries_;
};

} // namespace labrob