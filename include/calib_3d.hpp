////////////////////////////////////////////////////////////////////////////////
//
// Filename:      calib.hpp
// Authors:       Yu-Han, Hsueh
//
//////////////////////////////// FILE INFO /////////////////////////////////////
//
// calibration module for 3D-3D correspondences
//
/////////////////////////////////// LICENSE ////////////////////////////////////
//
// Copyright (C) 2020 Yu-Han, Hsueh <zero000.ece07g@nctu.edu.tw>
//
// This file is part of {calib_3d}.
//
//////////////////////////////////// NOTES /////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <vector>
#include <point3d.hpp>

namespace calib_3d {
// return the transformation from source frame to target frame
Pose Find_Transform_3D(const std::vector<Point3Data>& source,
                       const std::vector<Point3Data>& target,
                       const SE3& init_guess_transform);


// return the transformation from source frame to target frame
// certain parameters (x, y, z, yaw, pitch, roll) can be fixed if necessary
Pose Find_Transform_3D_Euler(const std::vector<Point3Data>& source,
                             const std::vector<Point3Data>& target,
                             const SE3& init_guess_transform,
                             const std::vector<int>& constant_indices);

Pose Find_Transform_3D_Diff(const std::vector<Point3Data>& source,
                            const std::vector<Point3Data>& target,
                            const SE3& init_guess_transform);

Pose Find_Transform_3D_Euler_Diff(const std::vector<Point3Data>& source,
                                  const std::vector<Point3Data>& target,
                                  const SE3& init_guess_transform,
                                  const std::vector<int>& constant_indices);

Pose Find_Transform_3D_Diff_resi(const std::vector<Point3Data>& source,
                                 const std::vector<Point3Data>& target,
                                 const SE3& init_guess_transform);

Pose Find_Transform_3D_Analytic(const std::vector<Point3Data>& source,
                                const std::vector<Point3Data>& target,
                                const SE3& init_guess_transform);

// Estimate average Euclidean error
double avg_error(const std::vector<Point3Data>& source,
                                 const std::vector<Point3Data>& target,
                                 const SE3& init_guess_transform);
}  // namespace calib_3d
