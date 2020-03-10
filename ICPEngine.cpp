#include "ICPEngine.h"

namespace ark {

	ICPEngine::ICPEngine(open3d::geometry::TriangleMesh model, Eigen::Matrix3d intr, int model_target_vertices, 
		int frame_downsample, Eigen::Matrix4d init, float rmse_threshold, 
		float fitness_threshold, float depth_threshold, float max_correspondence_dist) :
		model_target_vertices_(model_target_vertices), intr_inv_(intr.inverse()), frame_downsample_(frame_downsample),
		rmse_threshold_(rmse_threshold), fitness_threshold_(fitness_threshold), 
		depth_threshold_(depth_threshold), transform(init), max_correspondence_dist_(max_correspondence_dist) {

		//prepare source pointcloud
		model_pcld = open3d::geometry::PointCloud();
		convert_mesh_to_pointcloud(model, model_pcld);

		DisplayInitialization();

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
		initialized = false;
	}

	void ICPEngine::DisplayInitialization() {

		cout << "here1" << endl;
		cout << model_pcld.points_.size() << endl;
		cout << model_pcld.colors_.size() << endl;

		cv::namedWindow("image_init");
		cv::Mat bgr_init = cv::Mat(cv::Size(640, 480), CV_8UC3);

		auto iter_points = model_pcld.points_.begin();
		auto iter_colors = model_pcld.colors_.begin();

		Eigen::Matrix3d intr_ = intr_inv_.inverse();
		Eigen::Matrix4d w2c = transform.inverse();

		while (iter_points != model_pcld.points_.end() && iter_colors != model_pcld.colors_.end()) {

			Eigen::Vector3d v = *(iter_points++);
			Eigen::Vector3d c = *(iter_colors++);

			Eigen::Vector4d v_cam = w2c * (Eigen::Vector4d() << v(0), v(1), v(2), 1.0).finished();
			Eigen::Vector3d projected_pixel = intr_ * (Eigen::Vector3d() << v_cam(0), v_cam(1), v_cam(2)).finished();

			//point behind camera
			if (projected_pixel(2) <= 0) {
				continue;
			}

			projected_pixel /= projected_pixel(2);


			//outside frame
			if (projected_pixel(0) < 0 || projected_pixel(0) >= 640 || projected_pixel(1) < 0 || projected_pixel(1) >= 480) {
				continue;
			}

			//cout << projected_pixel << endl;
			//cout << c << endl;
			cv::Vec3b temp((int)round(c(2) * 255), (int)round(c(1) * 255), (int)round(c(0) * 255));
			bgr_init.at<cv::Vec3b>((int)projected_pixel(1), (int)projected_pixel(0)) = temp;
		}

		cout << "hi" << endl;

		cv::imshow("image_init", bgr_init);

	}

	void ICPEngine::MainLoop() {
		open3d::geometry::PointCloud frame_pcld = open3d::geometry::PointCloud();

		cout << "main looped" << endl;

		int counter = 1;

		while (!kill) {
			if (GetLatestTarget(frame_pcld)) {
				auto reg = open3d::registration::RegistrationColoredICP(frame_pcld, model_pcld, max_correspondence_dist_, transform,
					open3d::registration::ICPConvergenceCriteria(1e-6, 1e-6, 12));

				cout << reg.fitness_ << " " << reg.inlier_rmse_ << endl;

				if (reg.fitness_ > fitness_threshold_ && reg.inlier_rmse_ < rmse_threshold_ || (counter == 0 && reg.fitness_ > 0 && reg.inlier_rmse_ < rmse_threshold_)) {

					if (!initialized) {
						initialized = true;
						cv::destroyWindow("image_init");
					}

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

	void ICPEngine::PushFrame(MultiCameraFrame::Ptr frame) {

		cv::Mat depth;
		cv::Mat imRGB;

		frame->getImage(depth, 4);
		frame->getImage(imRGB, 3);

		open3d::geometry::PointCloud target = open3d::geometry::PointCloud();
		convert_depth_and_rgb_to_pointcloud_and_downsample(depth, imRGB, target, frame_downsample_, intr_inv_);
		target.EstimateNormals();
		frame_pcld = target;
	}

	void ICPEngine::ShutDown() {
		kill = true;
		icp.join();
	}

}