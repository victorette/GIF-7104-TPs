cmake_minimum_required (VERSION 2.6)

project (tp3Sequential)
 
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")

add_executable(tp1 main.cpp Matrix.cpp)
