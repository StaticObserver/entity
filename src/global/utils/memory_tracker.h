#ifndef MEMORY_TRACKER_H
#define MEMORY_TRACKER_H

#include <Kokkos_Core.hpp>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <mutex>
#include <cstdint>

// 内存分配跟踪器
class MemoryTracker {
public:
    static void allocate(const std::string &space, const std::string &name, void* ptr, uint64_t size) {
        std::lock_guard<std::mutex> lock(instance().mtx);
        instance().memoryMap[ptr] = {space, name, size};
    }

    static void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(instance().mtx);
        auto it = instance().memoryMap.find(ptr);
        if (it != instance().memoryMap.end()) {
            instance().memoryMap.erase(it);
        } else {
            std::cerr << "Warning: Attempting to free untracked memory! Pointer: " << ptr << std::endl;
        }
    }

    static void printMemoryUsage() {
        std::lock_guard<std::mutex> lock(instance().mtx);
        size_t totalAlloc = 0;
        std::cout << "Kokkos Profiling Memory Usage:" << std::endl;
        for (const auto& entry : instance().memoryMap) {
            std::cout << "Space: " << entry.second.space
                      << ", Label: " << entry.second.name
                      << ", Size: " << entry.second.size / 1024.0 << " KB" << std::endl;
            totalAlloc += entry.second.size;
        }
        std::cout << "Total allocated memory: " << totalAlloc / (1024.0 * 1024.0) << " MB" << std::endl;
    }

    static void saveMemoryUsageToFile(const std::string &filename, const std::string &projectName) {
        std::lock_guard<std::mutex> lock(instance().mtx);
        std::ofstream outfile(filename, std::ios::app);  // 以追加模式打开文件
        if (!outfile.is_open()) {
            std::cerr << "Error: Unable to open file for writing: " << filename << std::endl;
            return;
        }

        // 获取当前时间
        std::time_t now = std::time(nullptr);
        char timeBuffer[100];
        std::strftime(timeBuffer, sizeof(timeBuffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now));

        // 写入内存使用信息
        outfile << "Kokkos Profiling Memory Usage at " << timeBuffer;
        if (!projectName.empty()) {
            outfile << " [" << projectName << "]";
        }
        outfile << ":\n";

        size_t totalAlloc = 0;
        for (const auto &entry : instance().memoryMap) {
            outfile << "Space: " << entry.second.space
                    << ", Label: " << entry.second.name
                    << ", Size: " << entry.second.size / 1024.0 << " KB\n";
            totalAlloc += entry.second.size;
        }
        outfile << "Total allocated memory: " << totalAlloc / (1024.0 * 1024.0) << " MB\n";
        outfile << "-----------------------------------------\n";

        outfile.close();
    }

private:
    struct MemoryInfo {
        std::string space;
        std::string name;
        uint64_t size;
    };

    std::map<void*, MemoryInfo> memoryMap;
    std::mutex mtx;

    MemoryTracker() = default;

    static MemoryTracker& instance() {
        static MemoryTracker tracker;
        return tracker;
    }
};

// Profiling Hook：分配内存时被调用
extern "C" void kokkosp_allocate_data(Kokkos::Profiling::SpaceHandle handle, const char* name, void* ptr, uint64_t size) {
    MemoryTracker::allocate(handle.name, name ? name : "Unknown", ptr, size);
}

// Profiling Hook：释放内存时被调用
extern "C" void kokkosp_deallocate_data(Kokkos::Profiling::SpaceHandle handle, const char* name, void* ptr, uint64_t size) {
    MemoryTracker::deallocate(ptr);
}

#endif // MEMORY_TRACKER_H
