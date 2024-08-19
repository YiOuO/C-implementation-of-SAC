cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(sac)

find_package(Eigen3 REQUIRED)
set(CMAKE_PREFIX_PATH "/home/yi/libtorch/libtorch")
set(CMAKE_CXX_STANDARD 17)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(train_sac TrainSAC.cpp) 
target_link_libraries(train_sac ${TORCH_LIBRARIES})
target_include_directories(train_sac PRIVATE ${CMAKE_CURRENT_SOURCE})
set_property(TARGET train_sac PROPERTY CXX_STANDARD_17)

# add_executable(test_sac Testsac.cpp) 
# target_link_libraries(test_sac ${TORCH_LIBRARIES})
# target_include_directories(test_sac PRIVATE ${CMAKE_CURRENT_SOURCE})
# set_property(TARGET test_sac PROPERTY CXX_STANDARD_17)
