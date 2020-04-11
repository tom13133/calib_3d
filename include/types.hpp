#pragma once

#include <ostream>

#include <Eigen/Geometry>
#include <sophus/se3.hpp>

namespace calib_3d {
// use double for all geometrical types
typedef Eigen::AngleAxisd AngleAxis;
typedef Eigen::Quaterniond Quaternion;
typedef Eigen::Rotation2Dd Rotation2D;
typedef Eigen::Translation3d Translation;
typedef Eigen::Vector3d Vector3;
typedef Eigen::Matrix3d Matrix3;
typedef Quaternion Orientation;
typedef Vector3 Point;
typedef Sophus::SE3d SE3;
typedef Sophus::SO3d SO3;
typedef SE3 Pose;


inline double DegToRad(const double deg) {
  return deg * M_PI / 180.;
}


inline double RadToDeg(const double rad) {
  return rad * 180. / M_PI;
}
}  // namespace calib_3d
