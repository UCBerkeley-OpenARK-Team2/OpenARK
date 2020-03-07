#ifndef OPENARK_MESHIO_H
#define OPENARK_MESHIO_H

#include <mutex>
#include <thread>
#include <map>
#include <string>

#include "Open3D/IO/ClassIO/TriangleMeshIO.h"
#include "Open3D/IO/ClassIO/ImageIO.h"
#include "Open3D/IO/ClassIO/PointCloudIO.h"
#include "Open3D/Geometry/PointCloud.h"
#include <opencv2/opencv.hpp>
#include "Types.h"


namespace ark {
	//holds mesh

	open3d::geometry::TriangleMesh read_ply_file(std::string mesh_file);

	void write_ply_file(std::shared_ptr<open3d::geometry::TriangleMesh>, std::string file_name);

	void convert_mesh_to_pointcloud(open3d::geometry::TriangleMesh & tri_mesh, open3d::geometry::PointCloud & output);

	void convert_depth_to_pointcloud_and_downsample(cv::Mat depth, open3d::geometry::PointCloud & output, int stride, Eigen::Matrix3d proj_mat);

	void convert_depth_and_rgb_to_pointcloud_and_downsample(cv::Mat depth, cv::Mat rgb, open3d::geometry::PointCloud & output, int stride, Eigen::Matrix3d proj_mat);

	void downsample_pointcloud(open3d::geometry::PointCloud & output, int stride);

	void downsample_pointcloud_target(open3d::geometry::PointCloud & output, int target_vertices);

}

#endif