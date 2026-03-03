#ifndef VISC1_UDP_SENDER
#define VISC1_UDP_SENDER

#include <iostream>
#include <string>
#include <sstream>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "json.hpp"

// Header per Socket POSIX (Linux/macOS)
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>



class UdpSender {
private:
    std::string address_;
    int port_;
    int sock_;
    struct sockaddr_storage dest_addr_;
    socklen_t dest_addr_len_;
    double time_;

public:
    UdpSender(const std::string& address, int port);

    ~UdpSender();

    void sendJson(const nlohmann::json& j);
    
    template <typename T>
    void sendVector(const std::vector<std::string>& names, const std::vector<T>& values, const double timestamp, const std::string& timestamp_field = "timestamp") {
        if (names.size() != values.size()) {
            throw std::invalid_argument("Names and values vectors must have the same size.");
        }

        std::ostringstream json_stream;
        json_stream << "{";
        json_stream << "\"" << timestamp_field << "\": " << timestamp << ", ";
        for (size_t i = 0; i < names.size(); ++i) {
            json_stream << "\"" << names[i] << "\": " << values[i];
            if (i < names.size() - 1) {
                json_stream << ", ";
            }
        }
        json_stream << "}";

        send(json_stream);
    }
    void send(const std::string& json_string);
    void send(const std::ostringstream& json_stream);

};

#endif