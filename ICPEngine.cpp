#include "ICPEngine.h"

namespace ark {

	ICPEngine::ICPEngine(open3d::geometry::TriangleMesh model, int model_target_vertices, 
		int frame_downsample, Eigen::Matrix4d init, float rmse_threshold, 
		float fitness_threshold, float depth_threshold, float max_correspondence_dist) :
	model_target_vertices_(model_target_vertices), frame_downsample_(frame_downsample),
	rmse_threshold_(rmse_threshold), fitness_threshold_(fitness_threshold), 
		depth_threshold_(depth_threshold), transform(init), max_correspondence_dist_(max_correspondence_dist) {

		//prepare source pointcloud
		model_pcld = open3d::geometry::PointCloud();
		convert_mesh_to_pointcloud(model, model_pcld);
		downsample_pointcloud_target(model_pcld, model_target_vertices_);

		open3d::io::WritePointCloudToPLY("test.ply", model_pcld);

		max_iterations_ = 10;
		rmse_termination_ = 1e-6;
		fitness_termination_ = 1e-6;

		kill = false;
	}

	void ICPEngine::AddPoseAvailableHandler(PoseAvailableHandler handler, std::string handlerName) {
		callbacks[handlerName] = handler;
	}

	void ICPEngine::Start() {
		icp = std::thread(&ICPEngine::MainLoop, this);
	}

	void ICPEngine::MainLoop() {
		open3d::geometry::PointCloud frame_pcld = open3d::geometry::PointCloud();

		cout << "main looped" << endl;

		int counter = 0;

		while (!kill) {
			if (GetLatestTarget(frame_pcld)) {
				auto reg = open3d::registration::RegistrationColoredICP(frame_pcld, model_pcld, max_correspondence_dist_, transform,
					open3d::registration::ICPConvergenceCriteria(1e-6, 1e-6, 12));

				cout << reg.fitness_ << " " << reg.inlier_rmse_ << endl;

				if (reg.fitness_ > fitness_threshold_ && reg.inlier_rmse_ < rmse_threshold_ || (counter == 0 && reg.inlier_rmse_ < rmse_threshold_)) {
					transform = reg.transformation_;
					counter = 0;
					for (auto c: callbacks) {
						c.second(transform);
					}
				}
				else {
					counter++;
				}
			}
			else {
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
			}
		}

		cout << "main loop terminated" << endl;
	}

	bool ICPEngine::GetLatestTarget(open3d::geometry::PointCloud &target) {

		if (frameLock_.try_lock())
		{
			std::lock_guard<std::mutex> guard(frameLock_, std::adopt_lock);

			if (frame_pcld.points_.size() == 0) {
				return false;
			}

			target = frame_pcld;
			return true;
		}
	}

	void ICPEngine::PushFrame(MultiCameraFrame::Ptr frame, Eigen::Matrix3d intr_mat) {

		cv::Mat depth;
		cv::Mat imRGB;

		frame->getImage(depth, 4);
		frame->getImage(imRGB, 3);

		open3d::geometry::PointCloud target = open3d::geometry::PointCloud();
		convert_depth_and_rgb_to_pointcloud_and_downsample(depth, imRGB, target, frame_downsample_, intr_mat);
		target.EstimateNormals();
		frame_pcld = target;
	}

	void ICPEngine::ShutDown() {
		kill = true;
		icp.join();
	}

}