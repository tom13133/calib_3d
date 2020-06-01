#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <ros/package.h>
#include <ros/ros.h>

#include <calib_3d.hpp>
#include <point3d.hpp>

using calib_3d::Pose;
using calib_3d::Point;
using calib_3d::Orientation;
using calib_3d::Matrix3;
using calib_3d::Vector3;
using calib_3d::AngleAxis;

typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;

int main(int argc, char **argv) {
  ros::init(argc, argv, "calib_3d_node");
  ros::NodeHandle nh("~");

  std::vector<calib_3d::Point3Data> source_data;
  std::vector<calib_3d::Point3Data> target_data;

  // read data
  std::string path = ros::package::getPath("calib_3d");
  std::fstream file(path + "/dataset/correspondences.csv");

  if (file.good()) {
    std::string line;
    std::vector<double> pair_data;
    getline(file, line);
    double v;

    while (getline(file, line, '\n')) {
      pair_data.clear();
      std::istringstream templine(line);
      std::string data;
      while (getline(templine, data, ',')) {
        pair_data.push_back(atof(data.c_str()));
      }

      const calib_3d::Point3Data sd =
            calib_3d::Point3Data(std::make_shared<calib_3d::Point3D>(),
                             pair_data[0], pair_data[1], pair_data[2]);
      const calib_3d::Point3Data td =
            calib_3d::Point3Data(std::make_shared<calib_3d::Point3D>(),
                             pair_data[3], pair_data[4], pair_data[5]);
      source_data.push_back(sd);
      target_data.push_back(td);
    }
    std::cout << "Total " << source_data.size()
              << " data pair loaded." << std::endl;
  } else {
    std::cout << "Correspondences cannot be loaded." << std::endl;
    return 0;
  }

  // Given a initial guess
  std::vector<float> translation;
  std::vector<float> rotation;
  nh.getParam("/calib_3d_node/initial_guess/translation", translation);
  nh.getParam("/calib_3d_node/initial_guess/rotation", rotation);

  Point p_radar = Point(translation[0], translation[1], translation[2]);
  auto r = AngleAxis(calib_3d::DegToRad(rotation[0]), Vector3::UnitZ()) *
           AngleAxis(calib_3d::DegToRad(rotation[1]), Vector3::UnitY()) *
           AngleAxis(calib_3d::DegToRad(rotation[2]), Vector3::UnitX());
  Orientation o_radar = Orientation(r);
  Pose init_guess(o_radar, p_radar);
  Pose final_pose;

  double portion = 0;
  int times = 0;
  nh.getParam("/calib_3d_node/subsample/portion", portion);
  nh.getParam("/calib_3d_node/subsample/times", times);


  std::vector<calib_3d::Point3Data> source_data_sample;
  std::vector<calib_3d::Point3Data> target_data_sample;

  Matrix3 Rx;
  Vector3 euler;

  for (int iteration = 1; iteration <= times; iteration++) {
    std::cout << "Round: (" << iteration << "/" << times << ")"<< std::endl;

    // random-sample the dataset
    source_data_sample.clear();
    target_data_sample.clear();

    std::srand(unsigned(std::time(0)));
    std::vector<int> myvector, myvector2;
    for (int i = 0; i < source_data.size(); i++) {
      myvector.push_back(i);
    }
    std::random_shuffle(myvector.begin(), myvector.end());
    for (int i = 0; i < source_data.size() * portion; i++) {
      int loc = myvector[i];
      source_data_sample.push_back(source_data[loc]);
      target_data_sample.push_back(target_data[loc]);
    }
    std::cout << source_data_sample.size()
              << " pair correspondences used for optimization." << std::endl;

    // Given a initial guess
    std::vector<float> translation;
    std::vector<float> rotation;

    // Start Optimization
    // Case 1: Optimized by quaternion in rotation part (Numeric derivatives)
    std::vector<double> time_points;
    std::vector<Pose> result;
    time_points.push_back(ros::Time::now().toSec());
    final_pose = Find_Transform_3D(source_data_sample,
                                   target_data_sample,
                                   init_guess);
    result.push_back(final_pose);

    // Case 2: Optimized by euler angle in rotation part (Numeric derivatives)
    time_points.push_back(ros::Time::now().toSec());
    final_pose = Find_Transform_3D_Euler(source_data_sample,
                                         target_data_sample,
                                         init_guess,
                                         std::vector<int> {});
    result.push_back(final_pose);

    // Case 3: Optimized by quaternion in rotation part (Automatic Derivatives)
    time_points.push_back(ros::Time::now().toSec());
    final_pose = Find_Transform_3D_Diff(source_data_sample,
                                        target_data_sample,
                                        init_guess);
    result.push_back(final_pose);

    // Case 4: Optimized by euler angle in rotation (Automatic Derivatives)
    time_points.push_back(ros::Time::now().toSec());
    final_pose = Find_Transform_3D_Euler_Diff(source_data_sample,
                                              target_data_sample,
                                              init_guess,
                                              std::vector<int> {});
    result.push_back(final_pose);

    // Case 5: Optimized by quaternion in rotation part with multiple residual
    // (Automatic Derivatives)
    time_points.push_back(ros::Time::now().toSec());
    final_pose = Find_Transform_3D_Diff_resi(source_data_sample,
                                             target_data_sample,
                                             init_guess);
    result.push_back(final_pose);

    // Case 6: Optimized by quaternion in rotation (Analytics Derivatives)
    time_points.push_back(ros::Time::now().toSec());
    final_pose = Find_Transform_3D_Analytic(source_data_sample,
                                              target_data_sample,
                                              init_guess);
    result.push_back(final_pose);


    time_points.push_back(ros::Time::now().toSec());
    Rx = final_pose.unit_quaternion().toRotationMatrix();
    euler = Rx.eulerAngles(2, 1, 0);

    for (int i = 0; i < result.size(); i++) {
      Rx = result[i].unit_quaternion().toRotationMatrix();
      euler = Rx.eulerAngles(2, 1, 0);
      std::cout << "Case " << (i+1)
                << ": Current tf(deg): ("
                << result[i].translation().x() << ", "
                << result[i].translation().y() << ", "
                << result[i].translation().z() << ", "
                << calib_3d::RadToDeg(euler[0]) << ", "
                << calib_3d::RadToDeg(euler[1]) << ", "
                << calib_3d::RadToDeg(euler[2]) << ") -> ";

      std::cout << "Cost time: "
                << time_points[i+1] - time_points[i]
                << ", Avg error:"
                << avg_error(source_data_sample, target_data_sample, result[i]) << std::endl;
    }
  }

  std::cout << "Done." << std::endl;

  // Visualize on RVIZ
  ros::Publisher source_pub = nh.advertise<PointCloud>("source_points", 1, true);;
  ros::Publisher target_pub = nh.advertise<PointCloud>("target_points", 1, true);;
  ros::Publisher result_pub = nh.advertise<PointCloud>("result_points", 1, true);;

  std::string topic_frame = "map";

  PointCloud::Ptr source_pts(new PointCloud);
  source_pts->width    = source_data.size();
  source_pts->height   = 1;
  source_pts->is_dense = false;
  source_pts->points.resize(source_pts->width * source_pts->height);
  source_pts->header.frame_id = topic_frame;

  for (size_t i = 0; i < source_data.size(); i++) {
    Point pt;
    pt = source_data[i].point_;
    source_pts->points[i].x = pt.x();
    source_pts->points[i].y = pt.y();
    source_pts->points[i].z = pt.z();
  }

  PointCloud::Ptr target_pts(new PointCloud);
  target_pts->width    = target_data.size();
  target_pts->height   = 1;
  target_pts->is_dense = false;
  target_pts->points.resize(target_pts->width * target_pts->height);
  target_pts->header.frame_id = topic_frame;

  for (size_t i = 0; i < target_data.size(); i++) {
    Point pt;
    pt = target_data[i].point_;
    target_pts->points[i].x = pt.x();
    target_pts->points[i].y = pt.y();
    target_pts->points[i].z = pt.z();
  }

  PointCloud::Ptr result_pts(new PointCloud);
  pcl::transformPointCloud(*source_pts, *result_pts, final_pose.matrix());

  ros::Rate rate(1);
  while (ros::ok()) {
    source_pub.publish(source_pts);
    target_pub.publish(target_pts);
    result_pub.publish(result_pts);
    ros::spinOnce();
    rate.sleep();
  }
}
