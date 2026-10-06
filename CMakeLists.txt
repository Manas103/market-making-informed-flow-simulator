cmake_minimum_required(VERSION 3.16)
project(market_making_informed_flow_simulator CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(ENABLE_ASAN "Build with Address/UndefinedBehavior sanitizers" OFF)

add_library(mmsim INTERFACE)
target_include_directories(mmsim INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MSVC)
  target_compile_options(mmsim INTERFACE /O2 /W4 /EHsc)
else()
  target_compile_options(mmsim INTERFACE -O2 -Wall -Wextra -Wpedantic)
endif()

if(ENABLE_ASAN)
  add_compile_options(-fsanitize=address,undefined -g -O1 -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

add_executable(run_sessions src/run_sessions.cpp)
target_link_libraries(run_sessions PRIVATE mmsim)

add_executable(adapter_test tests/adapter_test.cpp)
target_link_libraries(adapter_test PRIVATE mmsim)
target_include_directories(adapter_test PRIVATE tests)

add_executable(pnl_identity_test tests/pnl_identity_test.cpp)
target_link_libraries(pnl_identity_test PRIVATE mmsim)
target_include_directories(pnl_identity_test PRIVATE tests)

add_executable(run_options_sessions src/run_options_sessions.cpp)
target_link_libraries(run_options_sessions PRIVATE mmsim)

add_executable(options_pricer_test tests/options_pricer_test.cpp)
target_link_libraries(options_pricer_test PRIVATE mmsim)
target_include_directories(options_pricer_test PRIVATE tests)

add_executable(options_pnl_identity_test tests/options_pnl_identity_test.cpp)
target_link_libraries(options_pnl_identity_test PRIVATE mmsim)
target_include_directories(options_pnl_identity_test PRIVATE tests)

enable_testing()
add_test(NAME adapter COMMAND adapter_test)
add_test(NAME pnl_identity COMMAND pnl_identity_test)
add_test(NAME options_pricer COMMAND options_pricer_test)
add_test(NAME options_pnl_identity COMMAND options_pnl_identity_test)
