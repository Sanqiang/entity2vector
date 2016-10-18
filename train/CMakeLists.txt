cmake_minimum_required(VERSION 3.6)
project(train)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(SYSTEM /usr/local/include) #for link h file
link_directories(/usr/local/lib) #for link implemented lib
#set(SOURCE_FILES sample/sparsehash.cpp)
#set(SOURCE_FILES sample/kbest.cpp)
#include_directories(SYSTEM /fast_text)
#link_directories(/fast_text)
#set(SOURCE_FILES fast_text/main.cc fast_text/args.cc fast_text/utils.cc fast_text/fasttext.cc fast_text/dictionary.cc fast_text/model.cc fast_text/matrix.cc fast_text/vector.cc)
set(SOURCE_FILES src/app.cpp src/data.cpp src/model.cpp src/util.cpp src/matrix.cpp src/vector.cpp src/args.cpp src/controller.cpp)

add_executable(train ${SOURCE_FILES})

target_link_libraries(train)