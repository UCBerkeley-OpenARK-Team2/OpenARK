#include "D435iCamera.h"
#include "OkvisSLAMSystem.h"
#include <iostream>
//#include <direct.h>
#include <thread>
#include <chrono>
#include <ctime>  
#include "glfwManager.h"
#include "Util.h"
#include "SaveFrame.h"
#include "ICPEngine.h"
#include "Types.h"
#include "Open3D/Integration/ScalableTSDFVolume.h"
#include "Open3D/Integration/MovingTSDFVolume.h"
#include "MeshUtil.h"
#include "Open3D/Registration/Registration.h"
#include "Open3D/Registration/ColoredICP.h"
#include <map>

using namespace ark;

int main(int argc, char **argv)
{

	if (argc != 3) {
		std::cerr << "Usage: ./" << argv[0] << " <mesh file> <augmentation file>" << std::endl
			<< "Args given: " << argc << std::endl;
		return -1;
	}

	google::InitGoogleLogging(argv[0]);

	cv::namedWindow("image", cv::WINDOW_AUTOSIZE);

	printf("Camera initialization started...\n");
	fflush(stdout);
	D435iCamera camera;
	camera.start();

	printf("Camera-IMU initialization complete\n");
	fflush(stdout);

	open3d::geometry::TriangleMesh model = read_ply_file(argv[1]);

	if (model.vertices_.size() == 0) {
		cout << "read mesh failed" << endl;
		return -1;
	}



	std::map<std::string, Eigen::Vector3d> augmentations;

	std::string name;
	double a, b, c;

	std::ifstream file(argv[2]);
	while (file >> name >> a >> b >> c) {
		Eigen::Vector3d temp;
		temp << a, b, c;
		augmentations.insert(std::pair<std::string, Eigen::Vector3d>(name, temp));
	}
	file.close();



	//start up gui thing
	if (!MyGUI::Manager::init())
	{
		fprintf(stdout, "Failed to initialize GLFW\n");
		return -1;
	}

	std::vector<float> intr = camera.getColorIntrinsics();
	MyGUI::ARCameraWindow ar_win("AR Viewer", 640 * 1.5, 480 * 1.5, GL_RGB, GL_UNSIGNED_BYTE, intr[0], intr[1], intr[2], intr[3], 0.01, 100);

	Eigen::Matrix4d temp = (Eigen::Matrix4d() << 0.999779, -0.0141999, 0.015498, -2.5046e-15, 
		-0.0141994, 0.0873891, 0.996073, 3.63987e-14,
		-0.0154985, -0.996073, 0.0871682, -1.81829e-14,
		0, 0, 0, 1).finished();

	Eigen::Affine3d current_t = Eigen::Affine3d::Identity();
	current_t.matrix() = temp;

	cout << temp << endl;

	ICPEngine * icp = new ICPEngine(model, 10000, 90, temp);

	PoseAvailableHandler poseHandler([&current_t](Eigen::Matrix4d transform) {

		cout << "POSE RECEIVED" << endl;
		cout << transform << endl;
		Eigen::Affine3d temp(transform);
		current_t = temp;
	});

	icp->AddPoseAvailableHandler(poseHandler, "pose'd");
	icp->Start();

	std::vector<float> intrinsics = camera.getColorIntrinsics();

	Eigen::Matrix3d intr_mat = (Eigen::Matrix3d() << intrinsics[0], 0, intrinsics[2], 0, intrinsics[1], intrinsics[3], 0, 0, 1).finished();
	intr_mat = intr_mat.inverse().eval();

	auto time = std::chrono::system_clock::now();
	
	while (MyGUI::Manager::running()) {

		//Update the display
		MyGUI::Manager::update();

		//Get current camera frame
		MultiCameraFrame::Ptr frame(new MultiCameraFrame);
		camera.update(*frame);
		icp->PushFrame(frame, intr_mat);

		cv::Mat imRGB;
		frame->getImage(imRGB, 3);
		ar_win.set_camera(current_t);
		ar_win.set_image(frame->images_[3]);


		int k = cv::waitKey(2);

		if (k == 'q' || k == 'Q' || k == 27) {
			break; // 27 is ESC
		}

		if (k == ' ') {

			for (auto pair : augmentations) {
				std::string cube_name = pair.first;
				Eigen::Matrix4d obj_pos = Eigen::Matrix4d::Identity();
				obj_pos.block<3, 3>(0, 0) = current_t.rotation();
				obj_pos.block<3, 1>(0, 3) = pair.second;
				Eigen::Affine3d pos(obj_pos);
				//Eigen::Affine3d pos = Eigen::Affine3d::Identity();
				MyGUI::Object* obj = new MyGUI::Cube(cube_name, 0.1, 0.1, 0.1);

				cout << pos.matrix() << endl;

				obj->set_transform(pos);
				//MyGUI::Augmentation* obj = new MyGUI::Augmentation(cube_name, 0.3, 0.3, 0.3, obj_pos);
				std::cout << "Adding cube " << cube_name << std::endl;
				ar_win.add_object(obj);
			}
		}

	}

	icp->ShutDown();

	cout << "terminating" << endl;

}