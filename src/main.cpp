#include <iostream>
#include <opencv2/opencv.hpp>
#include "vulkan/gpu.h"
#include "vulkan/vkmat.h"
#include "vulkan/option.h"
#include "layer/divide_vulkan.h"
#include "layer/resize_vulkan.h"
#include "layer/convolution_vulkan.h"
#include "layer/image_conv_vulkan.h"
#include "layer/image_conv.h"
#include "common.h"

using namespace iml::train;

void divide_forward(VulkanDevice* vkdev, Option& opt, cv::Mat& mat);
void resize_forward(VulkanDevice* vkdev, Option& opt, cv::Mat& mat);
void convolution_forward(VulkanDevice* vkdev, Option& opt, cv::Mat& mat);

void image_conv_test(VulkanDevice* vkdev, Option& opt, cv::Mat& mat);

void gpu_extract(VulkanDevice* vkdev, cv::Mat& mat) {
	std::cout << "start run vulkan" << std::endl;

	mat.convertTo(mat, CV_32FC3);

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

	//divide_forward(vkdev, opt, mat);
	//resize_forward(vkdev, opt, mat);
	//convolution_forward(vkdev, opt, mat);

	image_conv_test(vkdev, opt, mat);

	if (local_blob_allocator)
	{
		vkdev->reclaim_blob_allocator(local_blob_allocator);
		if (opt.workspace_vkallocator == opt.blob_vkallocator)
		{
			opt.workspace_vkallocator = 0;
		}
		opt.blob_vkallocator = 0;
	}
	if (local_staging_allocator)
	{
		vkdev->reclaim_staging_allocator(local_staging_allocator);
		opt.staging_vkallocator = 0;
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
