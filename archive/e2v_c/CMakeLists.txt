cmake_minimum_required(VERSION 3.5)
project(e2v)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(SOURCE_FILES e2v.c)

add_executable(e2v ${SOURCE_FILES})
target_link_libraries(e2v m pthread)
