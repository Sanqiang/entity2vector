cmake_minimum_required(VERSION 3.6)
project(train)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(SYSTEM /usr/local/include/json-c/) #for link h file
link_directories(/usr/local/lib) #for link implemented lib
set(SOURCE_FILES sample/test.c)
add_executable(train ${SOURCE_FILES})

target_link_libraries(train json-c)