#include <hrp4_locomotion/UdpSender.hpp>

UdpSender::UdpSender(const std::string& address, int port) 
        : address_(address), port_(port), sock_(-1), time_(0.0) {
        
        // Strutture per IPv4 e IPv6
        struct sockaddr_in addr4;
        struct sockaddr_in6 addr6;

        // Pulizia memoria
        std::memset(&dest_addr_, 0, sizeof(dest_addr_));

        // Proviamo a interpretare l'indirizzo come IPv4
        if (inet_pton(AF_INET, address_.c_str(), &addr4.sin_addr) == 1) {
            std::cout << "Opening IPv4 UDP socket for address " << address_ << " on port " << port_ << "..." << std::endl;
            sock_ = socket(AF_INET, SOCK_DGRAM, 0);
            
            addr4.sin_family = AF_INET;
            addr4.sin_port = htons(port_);
            
            std::memcpy(&dest_addr_, &addr4, sizeof(addr4));
            dest_addr_len_ = sizeof(addr4);

        // Se non è IPv4, proviamo come IPv6
        } else if (inet_pton(AF_INET6, address_.c_str(), &addr6.sin6_addr) == 1) {
            std::cout << "Opening IPv6 UDP socket for address " << address_ << " on port " << port_ << "..." << std::endl;
            sock_ = socket(AF_INET6, SOCK_DGRAM, 0);
            
            addr6.sin6_family = AF_INET6;
            addr6.sin6_port = htons(port_);
            
            std::memcpy(&dest_addr_, &addr6, sizeof(addr6));
            dest_addr_len_ = sizeof(addr6);

        } else {
            throw std::invalid_argument("Invalid IP address format.");
        }

        if (sock_ < 0) {
            throw std::runtime_error("Failed to create socket.");
        }
    }

UdpSender::~UdpSender() {
    if (sock_ != -1) {
        close(sock_);
    }
}

void UdpSender::send(const std::string& json_string) {
    sendto(sock_, json_string.c_str(), json_string.length(), 0, (struct sockaddr*)&dest_addr_, dest_addr_len_);
}

void UdpSender::send(const std::ostringstream& json_stream){
    send(json_stream.str());
}

void UdpSender::sendJson(const nlohmann::json& j) {
    std::string json_string = j.dump();
    send(json_string);
}