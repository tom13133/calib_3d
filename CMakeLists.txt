cmake_minimum_required(VERSION 2.8.3)
project(calib_3d)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
catkin_package()
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED
  roscpp
  roslib
  pcl_ros
  sensor_msgs
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}  
  ${SOPHUS_INCLUDE_DIRS}
)


## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/calib_3d.cpp
# )
## Declare a C++ library
 add_library(${PROJECT_NAME}
   src/calib_3d.cpp
   src/point3d.cpp
 )



add_executable(calib_3d_node scripts/calib_3d_node.cpp)
add_dependencies(calib_3d_node ${main_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(calib_3d_node ${catkin_LIBRARIES} ${PROJECT_NAME} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${SOPHUS_LIBRARIES})


