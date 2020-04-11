#include <point3d.hpp>

namespace calib_3d {
Point3D::Point3D(const int id, const std::string frame)
               : id_(id), frame_(frame) {}


Point3Data::Point3Data(Point3Ptr point3ptr,
                       const double x, const double y, const double z)
                       : point3ptr_(point3ptr), point_(x, y, z) {}


Point3Data::Point3Data(Point3Ptr point3ptr, const Point point)
                       : point3ptr_(point3ptr), point_(point) {}

}  // namespace calib_3d
