#include <thread>
#include <chrono>
#include <ctime>  
#include <map>
#include <mutex>
#include "Util.h"
#include "Types.h"
#include "MeshUtil.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Registration/Registration.h"
#include "Open3D/Registration/ColoredICP.h"

namespace ark {

	typedef std::function<void(Eigen::Matrix4d)> PoseAvailableHandler;

	class ICPEngine {
	public:
		ICPEngine(open3d::geometry::TriangleMesh model, int model_target_vertices = 10000, 
			int frame_downsample = 90, Eigen::Matrix4d init = Eigen::Matrix4d::Identity(), 
			float rmse_threshold = 0.05, float fitness_threshold = 0.8, float depth_threshold = 3.0,
			float max_correspondence_dist = 0.1);
		void Start();
		void PushFrame(MultiCameraFrame::Ptr frame, Eigen::Matrix3d intr_mat);
		//Eigen::Matrix4d GetPose();
		void AddPoseAvailableHandler(PoseAvailableHandler handler, std::string handlerName);
		void ShutDown();

		int model_target_vertices_;
		int frame_downsample_;

		float rmse_threshold_;
		float fitness_threshold_;
		float depth_threshold_;

		float max_correspondence_dist_;
		int max_iterations_;
		float rmse_termination_;
		float fitness_termination_;

	private:
		void MainLoop();
		bool GetLatestTarget(open3d::geometry::PointCloud &target);
		open3d::geometry::PointCloud model_pcld;
		open3d::geometry::PointCloud frame_pcld;
		
		std::map<std::string, PoseAvailableHandler> callbacks;

		bool kill;
		Eigen::Matrix4d transform;

		std::thread icp;
		std::mutex frameLock_;

	};




}