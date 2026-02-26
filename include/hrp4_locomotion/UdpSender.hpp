#ifndef VISC1_UDP_SENDER
#define VISC1_UDP_SENDER

#include <iostream>
#include <string>
#include <sstream>
#include <cstring>
#include <stdexcept>

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

    void send(const std::string& json_string);
    void send(const std::ostringstream& json_stream);

};

#endif