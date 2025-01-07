#ifndef MEOMORY_TRACKER_H
#define MEOMORY_TRACKER_H

#include <Kokkos_Core.hpp>
#include <iostream>
#include <map>
#include <string>
#include <mutex>

// 内存分配跟踪器
class MemoryTracker {
public:
    static void allocate(const std::string &label, size_t size, void *ptr) {
        std::lock_guard<std::mutex> lock(instance().mtx);
        instance().memoryMap[ptr] = {label, size};
    }

    static void deallocate(void *ptr) {
        std::lock_guard<std::mutex> lock(instance().mtx);
        if (instance().memoryMap.find(ptr) != instance().memoryMap.end()) {
            instance().memoryMap.erase(ptr);
        } else {
            std::cerr << "Warning: Attempting to free untracked memory!" << std::endl;
        }
    }

    static void printMemoryUsage() {
        std::lock_guard<std::mutex> lock(instance().mtx);
        size_t totalAlloc = 0;
        std::cout << "Kokkos Profiling Memory Usage:" << std::endl;
        for (const auto &entry : instance().memoryMap) {
            std::cout << "Label: " << entry.second.label
                      << ", Size: " << entry.second.size / 1024.0 << " KB" << std::endl;
            totalAlloc += entry.second.size;
        }
        std::cout << "Total allocated memory: " << totalAlloc / (1024.0 * 1024.0) << " MB" << std::endl;
    }

private:
    struct MemoryInfo {
        std::string label;
        size_t size;
    };

    std::map<void *, MemoryInfo> memoryMap;
    std::mutex mtx;

    MemoryTracker() = default;

    static MemoryTracker &instance() {
        static MemoryTracker tracker;
        return tracker;
    }
};

// Profiling 回调函数
void kokkos_allocate_callback(const char *label, const void *ptr, size_t size) {
    MemoryTracker::allocate(label ? label : "Unknown", size, const_cast<void *>(ptr));
}

void kokkos_deallocate_callback(const char *label, const void *ptr, size_t size) {
    MemoryTracker::deallocate(const_cast<void *>(ptr));
}



#endif // MEOMORY_TRACKER_H