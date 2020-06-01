#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <calib_3d.hpp>

namespace calib_3d {
// in-file util
static double Distance(const Point& p1, const Point& p2) {
  const double dx = p1.x() - p2.x();
  const double dy = p1.y() - p2.y();
  const double dz = p1.z() - p2.z();

  return sqrt(dx * dx + dy * dy + dz * dz);
}


class LocalParameterizationSE3 : public ceres::LocalParameterization {
 public:
  virtual ~LocalParameterizationSE3() {}

  // SE3 plus operation for Ceres
  //
  //  T * exp(x)
  //
  virtual bool Plus(double const* T_raw, double const* delta_raw,
                    double* T_plus_delta_raw) const {
    Eigen::Map<Sophus::SE3d const> const T(T_raw);
    Eigen::Map<Sophus::Vector6d const> const delta(delta_raw);
    Eigen::Map<Sophus::SE3d> T_plus_delta(T_plus_delta_raw);
    T_plus_delta = T * Sophus::SE3d::exp(delta);

    return true;
  }

  // Jacobian of SE3 plus operation for Ceres
  //
  // Dx T * exp(x)  with  x=0
  //
  virtual bool ComputeJacobian(double const* T_raw,
                               double* jacobian_raw) const {
    Eigen::Map<Sophus::SE3d const> T(T_raw);
    Eigen::Map<Eigen::Matrix<double, 6, 7> > jacobian(jacobian_raw);
    jacobian = T.Dx_this_mul_exp_x_at_0();
    return true;
  }

  virtual int GlobalSize() const { return Sophus::SE3d::num_parameters; }
  virtual int LocalSize() const { return Sophus::SE3d::DoF; }
};  // class LocalParameterizationSE3


class ErrorTerm_3D {
 public:
  ErrorTerm_3D(const Point& pt_1, const Point& pt_2)
      : pt_1_(pt_1), pt_2_(pt_2) {}

  bool operator()(const double* const pose_ptr, double* residual) const {
    Eigen::Map<SE3 const> const pose(pose_ptr);
    Point pt_1_transformed
          = pose * pt_1_;
    residual[0] = std::pow(Distance(pt_1_transformed, pt_2_), 2);

    return true;
  }

  static ceres::CostFunction* Create(const Point& pt_1,
                                     const Point& pt_2) {
    return new ceres::NumericDiffCostFunction<
        ErrorTerm_3D, ceres::RIDDERS, 1, SE3::num_parameters>(
        new ErrorTerm_3D(pt_1, pt_2));
  }

 private:
  const Point pt_1_;
  const Point pt_2_;
};  // class ErrorTerm_3D

// return SE3(computed transform)
Pose Find_Transform_3D(const std::vector<Point3Data>& source,
                       const std::vector<Point3Data>& target,
                       const SE3& init_guess_transform) {
  assert(source.size() == target.size() );
  ceres::Problem problem;

  SE3 pose = init_guess_transform;
  // Specify local update rule for our parameter
  problem.AddParameterBlock(pose.data(),
                            SE3::num_parameters,
                            new LocalParameterizationSE3);

  for (size_t i = 0; i < source.size(); ++i) {
    ceres::CostFunction* cost_function =
        ErrorTerm_3D::Create(source[i].point_, target[i].point_);
    problem.AddResidualBlock(cost_function, NULL, pose.data());
  }

  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  options.gradient_tolerance = 1e-6;  // * Sophus::Constants<double>::epsilon();
  options.function_tolerance = 1e-6;  // * Sophus::Constants<double>::epsilon();

  options.linear_solver_type = ceres::DENSE_QR;
  options.max_linear_solver_iterations = 200;
  options.max_num_iterations = 1000;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // Print result
  Matrix3 Rx = pose.unit_quaternion().toRotationMatrix();
  Vector3 euler = Rx.eulerAngles(2, 1, 0);
  Matrix3 init_Rx = init_guess_transform.unit_quaternion().toRotationMatrix();
  Vector3 init_euler = init_Rx.eulerAngles(2, 1, 0);
  std::cout << "init pose: (x, y, z, yaw, pitch, raw)(deg) = ("
            << init_guess_transform.translation().x() << ", "
            << init_guess_transform.translation().y() << ", "
            << init_guess_transform.translation().z() << ", "
            << RadToDeg(init_euler[0]) << ", " << RadToDeg(init_euler[1])
            << ", " << RadToDeg(init_euler[2]) << " )" << std::endl;
  std::cout << "final pose: (x, y, z, yaw, pitch, raw)(deg) = ("
            << pose.translation().x() << ", " << pose.translation().y() << ", "
            << pose.translation().z() << ", " << RadToDeg(euler[0]) << ", "
            << RadToDeg(euler[1]) << ", " << RadToDeg(euler[2]) << " )"
            << std::endl;

  std::cout << summary.BriefReport() << std::endl;

  return pose;
}


class ErrorTerm_3D_Euler {
 public:
  ErrorTerm_3D_Euler(const Point& pt_1, const Point& pt_2)
      : pt_1_(pt_1), pt_2_(pt_2) {}

  bool operator()(const double* const pose_ptr, double* residual) const {
    Point p_radar = Point(pose_ptr[0], pose_ptr[1], pose_ptr[2]);
    auto rotation = AngleAxis(pose_ptr[3], Vector3::UnitZ()) *
                    AngleAxis(pose_ptr[4], Vector3::UnitY()) *
                    AngleAxis(pose_ptr[5], Vector3::UnitX());
    Orientation o_radar = Orientation(rotation);
    Pose pose(o_radar, p_radar);

    Point pt_1_transformed = pose * pt_1_;
    residual[0] = std::pow(Distance(pt_1_transformed, pt_2_), 2);

    return true;
  }

  static ceres::CostFunction* Create(const Point& pt_1,
                                     const Point& pt_2) {
    return new ceres::NumericDiffCostFunction<ErrorTerm_3D_Euler, ceres::RIDDERS, 1, 6>(
        new ErrorTerm_3D_Euler(pt_1, pt_2));
  }

 private:
  const Point pt_1_;
  const Point pt_2_;
};  // class ErrorTerm_3D_Euler

// return SE3(computed transform)
Pose Find_Transform_3D_Euler(const std::vector<Point3Data>& source,
                             const std::vector<Point3Data>& target,
                             const SE3& init_guess_transform,
                             const std::vector<int>& constant_indices) {
  assert(source.size() == target.size());
  ceres::Problem problem;

  Matrix3 init_Rx = init_guess_transform.unit_quaternion().toRotationMatrix();
  Vector3 init_euler = init_Rx.eulerAngles(2, 1, 0);
  double pose_[] = {init_guess_transform.translation().x(),
                    init_guess_transform.translation().y(),
                    init_guess_transform.translation().z(),
                    init_euler[0],
                    init_euler[1],
                    init_euler[2]};

  // Specify local update rule for our parameter
  problem.AddParameterBlock(pose_, 6);

  // Fix certain parameters
  if (constant_indices.size() != 0) {
    ceres::SubsetParameterization* subset_parameterization
                      = new ceres::SubsetParameterization(6, constant_indices);
    problem.SetParameterization(pose_, subset_parameterization);
  }

  for (size_t i = 0; i < source.size(); ++i) {
    ceres::CostFunction* cost_function =
        ErrorTerm_3D_Euler::Create(source[i].point_, target[i].point_);
    problem.AddResidualBlock(cost_function, NULL, pose_);
  }

  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  options.gradient_tolerance = 1e-6;  // * Sophus::Constants<double>::epsilon();
  options.function_tolerance = 1e-6;  // * Sophus::Constants<double>::epsilon();

  options.linear_solver_type = ceres::DENSE_QR;
  options.max_linear_solver_iterations = 200;
  options.max_num_iterations = 1000;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  Point p_radar = Point(pose_[0], pose_[1], pose_[2]);
  auto rotation = AngleAxis(pose_[3], Vector3::UnitZ()) *
                  AngleAxis(pose_[4], Vector3::UnitY()) *
                  AngleAxis(pose_[5], Vector3::UnitX());
  Orientation o_radar = Orientation(rotation);
  Pose pose(o_radar, p_radar);

  // Print result
  Matrix3 Rx = pose.unit_quaternion().toRotationMatrix();
  Vector3 euler = Rx.eulerAngles(2, 1, 0);

  std::cout << "init pose: (x, y, z, yaw, pitch, raw)(deg) = ("
            << init_guess_transform.translation().x() << ", "
            << init_guess_transform.translation().y() << ", "
            << init_guess_transform.translation().z() << ", "
            << RadToDeg(init_euler[0]) << ", " << RadToDeg(init_euler[1])
            << ", " << RadToDeg(init_euler[2]) << " )" << std::endl;
  std::cout << "final pose: (x, y, z, yaw, pitch, raw)(deg) = ("
            << pose.translation().x() << ", " << pose.translation().y() << ", "
            << pose.translation().z() << ", " << RadToDeg(euler[0]) << ", "
            << RadToDeg(euler[1]) << ", " << RadToDeg(euler[2]) << " )"
            << std::endl;

  std::cout << summary.BriefReport() << std::endl;

  return pose;
}


class ErrorTerm_3D_Diff {
 public:
  ErrorTerm_3D_Diff(const Point& pt_1, const Point& pt_2)
      : pt_1_(pt_1), pt_2_(pt_2) {}
  template <typename T>
  bool operator()(const T* const pose_ptr, T* residual) const {
    using Vector3T = Eigen::Matrix<T, 3, 1>;
    Eigen::Map<Sophus::SE3<T> const> const pose(pose_ptr);

    Vector3T pt_1_transformed = pose * pt_1_.cast<T>();
    Vector3T pt_2 = pt_2_.cast<T>();
    residual[0] = ceres::pow((pt_1_transformed-pt_2).norm(), 2);
    return true;
  }

  static ceres::CostFunction* Create(const Point& pt_1,
                                     const Point& pt_2) {
    return new ceres::AutoDiffCostFunction<
        ErrorTerm_3D_Diff, 1, SE3::num_parameters>(
        new ErrorTerm_3D_Diff(pt_1, pt_2));
  }

 private:
  const Point pt_1_;
  const Point pt_2_;
};  // class ErrorTerm_3D_Diff

Pose Find_Transform_3D_Diff(const std::vector<Point3Data>& source,
                            const std::vector<Point3Data>& target,
                            const SE3& init_guess_transform) {
  assert(source.size() == target.size() );
  ceres::Problem problem;

  SE3 pose = init_guess_transform;
  // Specify local update rule for our parameter
  problem.AddParameterBlock(pose.data(),
                            SE3::num_parameters,
                            new LocalParameterizationSE3);

  for (size_t i = 0; i < source.size(); ++i) {
    ceres::CostFunction* cost_function =
        ErrorTerm_3D_Diff::Create(source[i].point_, target[i].point_);
    problem.AddResidualBlock(cost_function, NULL, pose.data());
  }

  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  options.gradient_tolerance = 1e-6;  // * Sophus::Constants<double>::epsilon();
  options.function_tolerance = 1e-6;  // * Sophus::Constants<double>::epsilon();

  options.linear_solver_type = ceres::DENSE_QR;
  options.max_linear_solver_iterations = 200;
  options.max_num_iterations = 1000;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // Print result
  Matrix3 Rx = pose.unit_quaternion().toRotationMatrix();
  Vector3 euler = Rx.eulerAngles(2, 1, 0);
  Matrix3 init_Rx = init_guess_transform.unit_quaternion().toRotationMatrix();
  Vector3 init_euler = init_Rx.eulerAngles(2, 1, 0);
  std::cout << "init pose: (x, y, z, yaw, pitch, raw)(deg) = ("
            << init_guess_transform.translation().x() << ", "
            << init_guess_transform.translation().y() << ", "
            << init_guess_transform.translation().z() << ", "
            << RadToDeg(init_euler[0]) << ", " << RadToDeg(init_euler[1])
            << ", " << RadToDeg(init_euler[2]) << " )" << std::endl;
  std::cout << "final pose: (x, y, z, yaw, pitch, raw)(deg) = ("
            << pose.translation().x() << ", " << pose.translation().y() << ", "
            << pose.translation().z() << ", " << RadToDeg(euler[0]) << ", "
            << RadToDeg(euler[1]) << ", " << RadToDeg(euler[2]) << " )"
            << std::endl;

  std::cout << summary.BriefReport() << std::endl;

  return pose;
}


class ErrorTerm_3D_Euler_Diff {
 public:
  ErrorTerm_3D_Euler_Diff(const Point& pt_1, const Point& pt_2)
      : pt_1_(pt_1), pt_2_(pt_2) {}
  template <typename T>
  bool operator()(const T* const pose_ptr, T* residual) const {
    using Vector3T = Eigen::Matrix<T, 3, 1>;
    using QuaternionT = Eigen::Quaternion<T>;
    Vector3T p_radar = Vector3T(pose_ptr[0], pose_ptr[1], pose_ptr[2]);
    auto rotation = Eigen::AngleAxis<T>(pose_ptr[3], Vector3T::UnitZ()) *
                    Eigen::AngleAxis<T>(pose_ptr[4], Vector3T::UnitY()) *
                    Eigen::AngleAxis<T>(pose_ptr[5], Vector3T::UnitX());
    QuaternionT o_radar = QuaternionT(rotation);
    Vector3T pt_1_transformed = rotation * pt_1_.cast<T>();
    pt_1_transformed += p_radar;

    residual[0] = T(ceres::pow((pt_1_transformed- pt_2_.cast<T>()).norm(), 2));
    return true;
  }

  static ceres::CostFunction* Create(const Point& pt_1,
                                     const Point& pt_2) {
    return new ceres::AutoDiffCostFunction<
        ErrorTerm_3D_Euler_Diff, 1, 6>(
        new ErrorTerm_3D_Euler_Diff(pt_1, pt_2));
  }

 private:
  const Point pt_1_;
  const Point pt_2_;
};  // class ErrorTerm_3D_Euler_Diff

Pose Find_Transform_3D_Euler_Diff(const std::vector<Point3Data>& source,
                                  const std::vector<Point3Data>& target,
                                  const SE3& init_guess_transform,
                                  const std::vector<int>& constant_indices) {
  assert(source.size() == target.size() );
  ceres::Problem problem;

  Matrix3 init_Rx = init_guess_transform.unit_quaternion().toRotationMatrix();
  Vector3 init_euler = init_Rx.eulerAngles(2, 1, 0);
  double pose_[] = {init_guess_transform.translation().x(),
                    init_guess_transform.translation().y(),
                    init_guess_transform.translation().z(),
                    init_euler[0],
                    init_euler[1],
                    init_euler[2]};

  // Specify local update rule for our parameter
  problem.AddParameterBlock(pose_, 6);

  // Fix certain parameters
  if (constant_indices.size() != 0) {
    ceres::SubsetParameterization* subset_parameterization
                      = new ceres::SubsetParameterization(6, constant_indices);
    problem.SetParameterization(pose_, subset_parameterization);
  }

  for (size_t i = 0; i < source.size(); ++i) {
    ceres::CostFunction* cost_function =
        ErrorTerm_3D_Euler_Diff::Create(source[i].point_, target[i].point_);
    problem.AddResidualBlock(cost_function, NULL, pose_);
  }

  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  options.gradient_tolerance = 1e-6;  // * Sophus::Constants<double>::epsilon();
  options.function_tolerance = 1e-6;  // * Sophus::Constants<double>::epsilon();

  options.linear_solver_type = ceres::DENSE_QR;
  options.max_linear_solver_iterations = 200;
  options.max_num_iterations = 1000;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  Point p_radar = Point(pose_[0], pose_[1], pose_[2]);
  auto rotation = AngleAxis(pose_[3], Vector3::UnitZ()) *
                  AngleAxis(pose_[4], Vector3::UnitY()) *
                  AngleAxis(pose_[5], Vector3::UnitX());
  Orientation o_radar = Orientation(rotation);
  Pose pose(o_radar, p_radar);

  // Print result
  Matrix3 Rx = pose.unit_quaternion().toRotationMatrix();
  Vector3 euler = Rx.eulerAngles(2, 1, 0);

  std::cout << "init pose: (x, y, z, yaw, pitch, raw)(deg) = ("
            << init_guess_transform.translation().x() << ", "
            << init_guess_transform.translation().y() << ", "
            << init_guess_transform.translation().z() << ", "
            << RadToDeg(init_euler[0]) << ", " << RadToDeg(init_euler[1])
            << ", " << RadToDeg(init_euler[2]) << " )" << std::endl;
  std::cout << "final pose: (x, y, z, yaw, pitch, raw)(deg) = ("
            << pose.translation().x() << ", " << pose.translation().y() << ", "
            << pose.translation().z() << ", " << RadToDeg(euler[0]) << ", "
            << RadToDeg(euler[1]) << ", " << RadToDeg(euler[2]) << " )"
            << std::endl;

  std::cout << summary.BriefReport() << std::endl;

  return pose;
}


class ErrorTerm_3D_Diff_resi {
 public:
  ErrorTerm_3D_Diff_resi(const Point& pt_1, const Point& pt_2)
      : pt_1_(pt_1), pt_2_(pt_2) {}
  template <typename T>
  bool operator()(const T* const pose_ptr, T* residual) const {
    using Vector3T = Eigen::Matrix<T, 3, 1>;
    Eigen::Map<Sophus::SE3<T> const> const pose(pose_ptr);

    Vector3T pt_1_transformed = pose * pt_1_.cast<T>();
    Vector3T pt_2 = pt_2_.cast<T>();
    residual[0] = ceres::pow((pt_1_transformed[0]-pt_2[0]), 2);
    residual[1] = ceres::pow((pt_1_transformed[1]-pt_2[1]), 2);
    residual[2] = ceres::pow((pt_1_transformed[2]-pt_2[2]), 2);
    return true;
  }

  static ceres::CostFunction* Create(const Point& pt_1,
                                     const Point& pt_2) {
    return new ceres::AutoDiffCostFunction<
        ErrorTerm_3D_Diff_resi, 3, SE3::num_parameters>(
        new ErrorTerm_3D_Diff_resi(pt_1, pt_2));
  }

 private:
  const Point pt_1_;
  const Point pt_2_;
};  // class ErrorTerm_3D_Diff_resi

Pose Find_Transform_3D_Diff_resi(const std::vector<Point3Data>& source,
                                 const std::vector<Point3Data>& target,
                                 const SE3& init_guess_transform) {
  assert(source.size() == target.size() );
  ceres::Problem problem;

  SE3 pose = init_guess_transform;
  // Specify local update rule for our parameter
  problem.AddParameterBlock(pose.data(),
                            SE3::num_parameters,
                            new LocalParameterizationSE3);

  for (size_t i = 0; i < source.size(); ++i) {
    ceres::CostFunction* cost_function =
        ErrorTerm_3D_Diff_resi::Create(source[i].point_, target[i].point_);
    problem.AddResidualBlock(cost_function, NULL, pose.data());
  }

  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  options.gradient_tolerance = 1e-6;  // * Sophus::Constants<double>::epsilon();
  options.function_tolerance = 1e-6;  // * Sophus::Constants<double>::epsilon();

  options.linear_solver_type = ceres::DENSE_QR;
  options.max_linear_solver_iterations = 200;
  options.max_num_iterations = 1000;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // Print result
  Matrix3 Rx = pose.unit_quaternion().toRotationMatrix();
  Vector3 euler = Rx.eulerAngles(2, 1, 0);
  Matrix3 init_Rx = init_guess_transform.unit_quaternion().toRotationMatrix();
  Vector3 init_euler = init_Rx.eulerAngles(2, 1, 0);
  std::cout << "init pose: (x, y, z, yaw, pitch, raw)(deg) = ("
            << init_guess_transform.translation().x() << ", "
            << init_guess_transform.translation().y() << ", "
            << init_guess_transform.translation().z() << ", "
            << RadToDeg(init_euler[0]) << ", " << RadToDeg(init_euler[1])
            << ", " << RadToDeg(init_euler[2]) << " )" << std::endl;
  std::cout << "final pose: (x, y, z, yaw, pitch, raw)(deg) = ("
            << pose.translation().x() << ", " << pose.translation().y() << ", "
            << pose.translation().z() << ", " << RadToDeg(euler[0]) << ", "
            << RadToDeg(euler[1]) << ", " << RadToDeg(euler[2]) << " )"
            << std::endl;

  std::cout << summary.BriefReport() << std::endl;

  return pose;
}

class SE3Parameterization : public ceres::LocalParameterization {
 public:
  virtual ~SE3Parameterization() {}
  virtual int GlobalSize() const {return 6;}
  virtual int LocalSize() const {return 6;}

  virtual bool ComputeJacobian(const double* x, double* jacobian) const {
    ceres::MatrixRef(jacobian, 6, 6) = ceres::Matrix::Identity(6, 6);
    return true;
  }

  virtual bool Plus(const double* x,
                    const double* delta,
                    double* x_plus_delta) const {
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> lie(x);
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> delta_lie(delta);

    Sophus::SE3d T = Sophus::SE3d::exp(lie);
    Sophus::SE3d delta_T = Sophus::SE3d::exp(delta_lie);
    Eigen::Matrix<double, 6, 1> x_plus_delta_lie = (delta_T * T).log();

    for (int i = 0; i < 6; i++)
      x_plus_delta[i] = x_plus_delta_lie(i, 0);

    return true;
  }
};  // SE3Parameterization

class ErrorTerm_3D_Analytic: public ceres::SizedCostFunction<3, 6> {
 public:
  ErrorTerm_3D_Analytic(const Point& pt_1, const Point& pt_2)
                         : pt_1_(pt_1), pt_2_(pt_2) {}
  virtual ~ErrorTerm_3D_Analytic() {}
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> lie(*parameters);
    Pose pose = Sophus::SE3d::exp(lie);
    Point pt_1_transformed = pose * pt_1_;

    if (jacobians != NULL && jacobians[0] != NULL) {
      Eigen::Matrix<double, 3, 6> J;
      J << 1, 0, 0, 0, +pt_1_transformed.z(), -pt_1_transformed.y(),
           0, 1, 0, -pt_1_transformed.z(), 0, +pt_1_transformed.x(),
           0, 0, 1, +pt_1_transformed.y(), -pt_1_transformed.x(), 0;

      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 6; j++) {
          jacobians[0][i*6 + j] = J(i, j);
        }
      }
    }

    residuals[0] = pt_1_transformed.x() - pt_2_.x();
    residuals[1] = pt_1_transformed.y() - pt_2_.y();
    residuals[2] = pt_1_transformed.z() - pt_2_.z();

    return true;
  }

 private:
  const Point pt_1_;
  const Point pt_2_;
};  // ErrorTerm_3D_Analytic

Pose Find_Transform_3D_Analytic(const std::vector<Point3Data>& source,
                                const std::vector<Point3Data>& target,
                                const SE3& init_guess_transform) {
  assert(source.size() == target.size() );
  ceres::Problem problem;

  SE3 pose = init_guess_transform;
  Sophus::Vector6d se3 = pose.log();
  // Specify local update rule for our parameter

  double pose_[] = {se3[0], se3[1], se3[2], se3[3], se3[4], se3[5]};
  problem.AddParameterBlock(pose_, 6, new SE3Parameterization);

  for (size_t i = 0; i < source.size(); ++i) {
    ceres::CostFunction* cost_function
               = new ErrorTerm_3D_Analytic(source[i].point_, target[i].point_);
    problem.AddResidualBlock(cost_function, NULL, pose_);
  }

  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  options.gradient_tolerance = 1e-6;  // * Sophus::Constants<double>::epsilon();
  options.function_tolerance = 1e-6;  // * Sophus::Constants<double>::epsilon();

  options.linear_solver_type = ceres::DENSE_QR;
  options.max_linear_solver_iterations = 200;
  options.max_num_iterations = 1000;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  se3 << pose_[0], pose_[1], pose_[2], pose_[3], pose_[4], pose_[5];
  pose = Sophus::SE3d::exp(se3);

  // Print result
  Matrix3 Rx = pose.unit_quaternion().toRotationMatrix();
  Vector3 euler = Rx.eulerAngles(2, 1, 0);
  Matrix3 init_Rx = init_guess_transform.unit_quaternion().toRotationMatrix();
  Vector3 init_euler = init_Rx.eulerAngles(2, 1, 0);
  std::cout << "init pose: (x, y, z, yaw, pitch, raw)(deg) = ("
            << init_guess_transform.translation().x() << ", "
            << init_guess_transform.translation().y() << ", "
            << init_guess_transform.translation().z() << ", "
            << RadToDeg(init_euler[0]) << ", " << RadToDeg(init_euler[1])
            << ", " << RadToDeg(init_euler[2]) << " )" << std::endl;
  std::cout << "final pose: (x, y, z, yaw, pitch, raw)(deg) = ("
            << pose.translation().x() << ", " << pose.translation().y() << ", "
            << pose.translation().z() << ", " << RadToDeg(euler[0]) << ", "
            << RadToDeg(euler[1]) << ", " << RadToDeg(euler[2]) << " )"
            << std::endl;

  std::cout << summary.BriefReport() << std::endl;

  return pose;
}

double avg_error(const std::vector<Point3Data>& source,
                 const std::vector<Point3Data>& target,
                 const SE3& pose) {
  double sum = 0;
  for (size_t i = 0; i < source.size(); ++i) {
    sum += ((pose * source[i].point_) - target[i].point_).norm();
  }
  return sum / source.size();
}
}  // namespace calib_3d
