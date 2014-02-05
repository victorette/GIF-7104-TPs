cmake_minimum_required (VERSION 2.6)

project (tp2)
 
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fopenmp -std=c++11")

add_executable(tp2 premiers.cpp)
