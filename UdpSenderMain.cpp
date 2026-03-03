#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <cmath>
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
    UdpSender(const std::string& address, int port) 
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

    ~UdpSender() {
        if (sock_ != -1) {
            close(sock_);
        }
    }

    void run() {
        // La stringa statica definita nel tuo script Python
        std::string test_str = R"({ "1252": { "timestamp": { "microsecond": 0 }, "value": { "current": { "ampere": null }, "voltage": { "volt": 24.852617263793945 } } } })";

        while (true) {
            // sleep(0.05)
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            time_ += 0.05;

            // Generazione JSON manuale (simula json.dumps)
            std::ostringstream json_stream;
            json_stream << "{\"timestamp\": " << time_ 
                        << ", \"test_data\": {\"cos\": " << std::cos(time_) 
                        << ", \"sin\": " << std::sin(time_) << "}}";
            
            std::string data = json_stream.str();

            // Invio del primo pacchetto
            sendto(sock_, data.c_str(), data.length(), 0, 
                  (struct sockaddr*)&dest_addr_, dest_addr_len_);

            // Invio del secondo pacchetto
            sendto(sock_, test_str.c_str(), test_str.length(), 0, 
                  (struct sockaddr*)&dest_addr_, dest_addr_len_);
        }
    }
};

// Funzione principale che imita argparse
int main(int argc, char* argv[]) {
    std::string address = "127.0.0.1";
    int port = 9870;

    // Parsing degli argomenti da riga di comando
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--address" && i + 1 < argc) {
            address = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        }
    }

    try {
        UdpSender sender(address, port);
        sender.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}