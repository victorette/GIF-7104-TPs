cmake_minimum_required (VERSION 2.6)

project (tp1)
 
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -lpthread -std=c++11")

add_executable(tp1 premiers_pthread.cpp)
