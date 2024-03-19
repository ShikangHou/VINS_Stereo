cmake_minimum_required(VERSION 3.0.2)
project(vins_stereo)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")


find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  image_transport
  roscpp
  std_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
  cv_bridge

)

find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem program_options system)
###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES VINS_Stereo
#  CATKIN_DEPENDS geometry_msgs image_transport roscpp std_msgs tf2 tf2_geometry_msgs tf2_ros
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(lib_vins
   src/feature_tracker/feature_tracker.cpp
   src/parameters.cpp
)
target_link_libraries(lib_vins
    ${OpenCV_LIBS} 
    ${CERES_LIBRARIES}
)

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(vins_node src/vins_node.cpp)


## Specify libraries to link a library or executable target against
target_link_libraries(vins_node
  ${catkin_LIBRARIES}
  lib_vins
  Threads::Threads
)
