#include "src/global/enums.h"
#include <iostream>

int main() {
    try {
        // 测试所有有效的枚举值
        auto boris = ntt::PrtlPusher::pick("boris");
        auto vay = ntt::PrtlPusher::pick("vay");
        auto photon = ntt::PrtlPusher::pick("photon");
        auto forcefree = ntt::PrtlPusher::pick("forcefree");
        auto none = ntt::PrtlPusher::pick("none");
        
        std::cout << "所有枚举值测试通过！" << std::endl;
        std::cout << "boris: " << boris.to_string() << std::endl;
        std::cout << "vay: " << vay.to_string() << std::endl;
        std::cout << "photon: " << photon.to_string() << std::endl;
        std::cout << "forcefree: " << forcefree.to_string() << std::endl;
        std::cout << "none: " << none.to_string() << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
} 