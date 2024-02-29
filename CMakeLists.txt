cmake_minimum_required(VERSION 3.26)

project(Istogramma)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp ")
set(CMAKE_EXE_LINKER_FLAGS "-static")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

add_executable(Istogramma main.cpp)




