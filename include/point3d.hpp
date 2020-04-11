#pragma once

#include <memory>
#include <string>
#include <types.hpp>

namespace calib_3d {
class Point3D {
 public:
  explicit Point3D(const int id = -1, const std::string frame = "map");
  inline virtual ~Point3D() {}

  inline int getId() { return id_; }
  inline std::string getFrame() { return frame_; }

 private:
  int id_;
  std::string frame_;
};  // class Point3D

using Point3Ptr = std::shared_ptr<Point3D>;

struct Point3Data {
  Point3Data(Point3Ptr point3ptr = std::make_shared<Point3D>(),
             Point point = Point());
  Point3Data(Point3Ptr point3ptr,
             const double x, const double y, const double z);

  Point3Ptr point3ptr_;
  Point point_;
};  // struct Point3Data
}  // namespace calib_3d
