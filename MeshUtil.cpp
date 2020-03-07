#include "MeshUtil.h"

namespace ark {
	void write_ply_file(std::shared_ptr<open3d::geometry::TriangleMesh> write_mesh, std::string file_name) {
		open3d::io::WriteTriangleMeshToPLY(file_name, *write_mesh, false, false, true, true, false, false);
	}

	open3d::geometry::TriangleMesh read_ply_file(std::string mesh_file) {

		auto mesh = open3d::geometry::TriangleMesh();

		if (!open3d::io::ReadTriangleMeshFromPLY(mesh_file, mesh, false)) {
			cout << "read mesh failed" << endl;
			return mesh;
		}

		cout << "read mesh " << mesh.vertices_.size() << " " << mesh.triangles_.size() << " " << mesh.vertex_normals_.size() << endl;

		return mesh;
	}

	//convert triangle mesh to pointcloud, uses triangle mesh vertices and vertex colors, averages adjacent triangles to find vertex normals
	void convert_mesh_to_pointcloud(open3d::geometry::TriangleMesh & tri_mesh, open3d::geometry::PointCloud & output) {

		cout << "converting mesh to pointcloud" << endl;

		cout << tri_mesh.vertices_.size() << endl;
		cout << tri_mesh.triangles_.size() << endl;

		output.colors_ = tri_mesh.vertex_colors_;
		output.points_ = tri_mesh.vertices_;
		output.normals_ = tri_mesh.vertex_normals_;
	}

	void convert_depth_to_pointcloud_and_downsample(cv::Mat depth, open3d::geometry::PointCloud & output, int stride, Eigen::Matrix3d proj_mat) {

		//staggering?

		int counter = 0;

		for (int i = 0; i < depth.rows; ++i) {
			for (int k = counter % stride; k < depth.cols; k += stride) {
				Eigen::Vector3d proj_pos(k, i, 1.0);
				proj_pos = proj_mat * proj_pos;
				proj_pos *= depth.at<uint16_t>(i, k) / 1000.0;

				//cout << proj_pos << endl;
				counter++;
				output.points_.push_back(proj_pos);
			}
		}
	}

	void convert_depth_and_rgb_to_pointcloud_and_downsample(cv::Mat depth, cv::Mat rgb, open3d::geometry::PointCloud & output, int stride, Eigen::Matrix3d proj_mat) {

		//staggering?
		int counter = 0;

		for (int i = 0; i < depth.rows; ++i) {
			for (int k = counter % stride; k < depth.cols; k += stride) {
				Eigen::Vector3d proj_pos(k, i, 1.0);

				float pix_depth = depth.at<uint16_t>(i, k) / 1000.0;

				proj_pos = proj_mat * proj_pos;
				proj_pos *= pix_depth;

				cv::Vec3b color = rgb.at<cv::Vec3b>(i, k);

				Eigen::Vector3d colorf(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0);

				//cout << proj_pos << endl;

				output.points_.push_back(proj_pos);
				output.colors_.push_back(colorf);

				counter++;
			}
		}

	}

	void downsample_pointcloud(open3d::geometry::PointCloud & output, int stride) {
		int idx = 0;

		std::vector<Eigen::Vector3d> new_pts;
		std::vector<Eigen::Vector3d> new_colors;
		std::vector<Eigen::Vector3d> new_normals;

		while (idx < output.points_.size()) {
			new_pts.push_back(output.points_[idx]);
			idx += stride;
		}

		idx = 0;
		while (idx < output.colors_.size()) {
			new_colors.push_back(output.colors_[idx]);
			idx += stride;
		}

		idx = 0;
		while (idx < output.normals_.size()) {
			new_normals.push_back(output.normals_[idx]);
			idx += stride;
		}


		output.points_ = new_pts;
		output.colors_ = new_colors;
		output.normals_ = new_normals;
	}

	void downsample_pointcloud_target(open3d::geometry::PointCloud & output, int target_vertices) {
		int orig_size = output.points_.size();
		int stride = max(orig_size / target_vertices, 1);

		cout << "stride " << stride << endl;

		downsample_pointcloud(output, stride);
	}
}
