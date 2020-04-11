# ROS package to estimate the transformation between 3D-3D correspondences.

The package is used to compute the transformation between 3D-3D correspondences.  

## Dependencies
1. Eigen3
2. Ceres Solver
3. Sophus
4. PCL

## Nodes
1. Node **calib_3d_node**

## 1. Node calib_3d_node
This node is used to estimate the 6 degree of freedom (DoF) transformation from 3D-3D correspondences.  

### (a) Dataset
This node reads dataset containnig 3D-3D correspondences.  
The dataset shall follow the format as follow:  

1. path:  
Its path is ```~/calib_3d/dataset/correspondences.csv```

2. format:
#### data.csv
> s_x, s_y, s_z, t_x, t_y, t_z  
> 5.88812, 5.02482, 0.521771, 7.57812, 0.957031, 0.380391  
> 7.81667, 3.53743, 0.57776, 8.35352, -1.32227, 0.400391  
> 4.49032, 0.365056, 0.148313, 3.71875, -2.14844, 0.0390625  
> ...  

Each correspondence includes (s_x, s_y, s_z, t_x, t_y, t_z),  
where **(s_x, s_y, s_z)** is the source point, and **(t_x, t_y, t_z)** is its corresponding target point.  
Notice that the first line in **data.csv** will be ignored.  

### (b) Setup
Before start running the calibration module, there is one configuration file that need to be specfied.  
We use ROS Parameter Server to set the configurations.  
1. path:  
Its path is ```~/calib_3d/config/calib_3d.yaml```

2. format:
#### lidar_ti_calibrate.yaml
> initial_guess:  
>   translation: [0, 0, 0]  
>   rotation: [20,0,0]  
> subsample:  
>   portion: 1  
>   times: 10  

where parameter **initial_guess** is needed to be specified, which contains translation (x, y, z) and rotation (yaw, pitch, roll),  
**subsmample/portion** determines how many correspondences used in each optimization, ranging from (0, 1).  
**subsmample/times** determines how many runs performing optimization.  

### (c) Getting Started.
```
roslaunch calib_3d calib_3d.launch
```

After running the node, it will publish the source points, target points, and calibration result points on RVIZ to show the performance of the module.  

* Result:  
Red: source points  
Green: target points  
White: result points (transformed source points)  
<img src="https://github.com/tom13133/calib_3d/blob/master/images/calib_result.png" width="800">
