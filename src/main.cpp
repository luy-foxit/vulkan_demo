#include <iostream>
#include "gpu.h"

using namespace iml::train;

void run_gpu(const char* image) {

}

int main(int argc, char* argv) {

	// init vulkan
	int ret = create_gpu_instance();
	if (ret) {
		std::cout << "create_gpu_instance error:" << std::endl;
		return ret;
	}


	// destroy vulkan
	destroy_gpu_instance();

	std::cout << "vulkan demo end." << std::endl;
	return 0;
}