#include "MLVSS.h"

int main() {
	SDKController::Initialize();
	torch::jit::Module Model = PyTorch::Load(Variables::PATH + "Model.pt");

	cv::Mat Frame;
	cv::Mat Screenshot;
	double CurrentTime = 0.0;
	double StartTime = 0.0;
	double EndTime = 0.0;
	while (true) {
		CurrentTime = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count() / 1000000000.0;
		StartTime = CurrentTime;

		ScreenCapture::TrackWindow("Truck Simulator", {"Discord"}, 2);

		Frame = ScreenCapture::GetLatestFrame();
		cv::imshow("Frame", Frame);
		cv::waitKey(1);

		EndTime = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count() / 1000000000.0;

		std::cout << "FPS: " << 1.0 / (EndTime - StartTime) << std::endl;
	}

	if (Variables::BUILD_TYPE == "Release") {
		system("pause");
	}
}