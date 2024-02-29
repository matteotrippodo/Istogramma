cmake_minimum_required(VERSION 3.26)
# Dichiarazione del percorso di Vcpkg
#set(CMAKE_TOOLCHAIN_FILE "C:/Users/matte/.vcpkg-clion/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
project(Istogramma)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp ")
set(CMAKE_EXE_LINKER_FLAGS "-static")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
#set(CMAKE_BUILD_TYPE Release)
#cmake_policy(SET CMP0144 NEW)

#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Zi")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -debug")
#include_directories(C:/MinGW/include)
#link_directories(C:/MinGW/lib)

add_executable(Istogramma main.cpp)
#set(BOOST_ROOT "C:/boost_1_84_0")
#find_package(Boost REQUIRED COMPONENTS filesystem)
#include_directories(${Boost_INCLUDE_DIRS})
#target_link_libraries(Istogramma ${Boost_LIBRARIES})
#set(BOOST_ROOT "C:/boost_1_84_0")
#set(BOOST_LIBRARYDIR "C:/boost_1_84_0/libs")
#find_package(Boost REQUIRED COMPONENTS filesystem)
#include_directories(${Boost_INCLUDE_DIRS})
#target_link_libraries(Istogramma ${Boost_LIBRARIES})
#find_package(Boost REQUIRED)
# Includi le directory di inclusione di Boost
#target_include_directories(Istogramma PRIVATE ${Boost_INCLUDE_DIRS})

# Linka Boost al tuo programma
#target_link_libraries(Istogramma PRIVATE ${Boost_LIBRARIES})

#find_package(OpenMP REQUIRED)
#if(OpenMP_CXX_FOUND)
#    target_link_libraries(Istogramma PUBLIC OpenMP::OpenMP_CXX)
#endif()


#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mavx")



