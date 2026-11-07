cmake_minimum_required(VERSION 3.16)
project(pymm CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(pymm binding.cpp)
target_include_directories(pymm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
