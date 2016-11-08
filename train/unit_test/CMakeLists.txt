cmake_minimum_required(VERSION 3.6)
project(test)

add_subdirectory(lib/gtest-1.7.0)
add_subdirectory(basic_test)



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

include_directories(SYSTEM /usr/local/include) #for link h file
link_directories(/usr/local/lib) #for link implemented lib

set(SOURCE_FILES test_serialize.cpp ../src/data.cpp ../src/model.cpp ../src/util.cpp ../src/matrix.cpp ../src/vector.cpp ../src/args.cpp ../src/controller.cpp)

add_executable(test ${SOURCE_FILES})

target_link_libraries(test)