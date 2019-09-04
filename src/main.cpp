#include <iostream>
#include "gpu.h"

using namespace iml::train;

int main(int argc, char* argv) {

	// init vulkan
	int ret = create_gpu_instance();
	if (ret) {
		std::cout << "create_gpu_instance error:" << std::endl;
		return ret;
	}

	VulkanDevice* vkdev = get_gpu_device();


	// destroy vulkan
	destroy_gpu_instance();

	std::cout << "vulkan demo end." << std::endl;
	return 0;
}