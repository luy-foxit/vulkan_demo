#include <iostream>
#include <opencv2/opencv.hpp>
#include "gpu.h"
#include "vkmat.h"
#include "option.h"
#include "layer/relu_100_vulkan.h"

using namespace iml::train;

void gpu_extract(VulkanDevice* vkdev, cv::Mat& mat) {
	std::cout << "start run vulkan" << std::endl;

	mat.convertTo(mat, CV_32FC3);
	
	ReLU_vulkan relu;
	int ret = relu.create_pipeline(vkdev);
	if (ret) {
		std::cout << "create_pipeline err:" << ret << std::endl;
		return;
	}

	//--TODO: 将weight, bias 拷贝到显存, relu不需要
	
	VkAllocator* local_blob_allocator = 0;
	VkAllocator* local_staging_allocator = 0;
	Option opt;
	if (!opt.blob_vkallocator) {
		local_blob_allocator = vkdev->acquire_blob_allocator();
		opt.blob_vkallocator = local_blob_allocator;
	}
	if (!opt.staging_vkallocator)
	{
		local_staging_allocator = vkdev->acquire_staging_allocator();
		opt.staging_vkallocator = local_staging_allocator;
	}

	VkCompute cmd(vkdev);
	VkMat vkmat;
	vkmat.create_like(mat, opt.blob_vkallocator, opt.staging_vkallocator);
	ret = relu.forward_inplace(vkmat, cmd);
	if (ret) {
		std::cout << "relu forward_inplace failed" << std::endl;
		return;
	}
}

int main(int argc, char* argv[]) {
	if (argc != 2)
	{
		fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
		return -1;
	}
	const char* image = argv[1];
	cv::Mat mat = cv::imread(image, cv::IMREAD_COLOR);

	// init vulkan
	int ret = create_gpu_instance();
	if (ret) {
		std::cout << "create_gpu_instance error:" << std::endl;
		return ret;
	}

	VulkanDevice* vkdev = get_gpu_device();		//获取vulkan逻辑设备
	gpu_extract(vkdev, mat);

	// destroy vulkan
	destroy_gpu_instance();

	std::cout << "vulkan demo end." << std::endl;
	return 0;
}