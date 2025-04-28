cmake_minimum_required(VERSION 3.20)
project(MPI_Project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED)

add_executable(${PROJECT_NAME} main.cc )

target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_CXX)

if(WIN32)
    target_include_directories(${PROJECT_NAME} PRIVATE "C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
    target_link_directories(${PROJECT_NAME} PRIVATE "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64")
endif()