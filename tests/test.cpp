#include "test_src.h"

// test external libraries
#include "test_extern_toml.h"
#include "test_extern_kokkos.h"

#include <acutest/acutest.h>

void testSuccess() {}

TEST_LIST = {{"lib/aux", testSrc},
             {"extern/toml", testExternToml},
             {"extern/kokkos", testExternKokkos},
             {"success", testSuccess},
             {nullptr, nullptr}};
