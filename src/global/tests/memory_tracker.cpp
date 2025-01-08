#include "utils/memory_tracker.h"

#include <Kokkos_Core.hpp>

#include "global.h"
#include "arch/kokkos_aliases.h"

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  Kokkos::Profiling::pushRegion("test");


  {
        // 在 Kokkos 中分配内存
        array_t<real_t *> view {"test", 10000};

        // 打印内存使用情况
        MemoryTracker::printMemoryUsage();
        MemoryTracker::saveMemoryUsageToFile("memory_usage.log", "allocate");

    }

  // 再次打印内存使用情况（释放后）
  MemoryTracker::printMemoryUsage();
  MemoryTracker::saveMemoryUsageToFile("memory_usage.log", "cleanup");
  
  Kokkos::Profiling::popRegion();
  Kokkos::finalize();

  return 0;
}