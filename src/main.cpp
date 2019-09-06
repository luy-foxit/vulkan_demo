#include <iostream>
#include <opencv2/opencv.hpp>
#include "gpu.h"
#include "vkmat.h"
#include "option.h"
#include "layer/relu_100_vulkan.h"
#include "layer/resize_vulkan.h"

using namespace iml::train;

void relu_forward(VulkanDevice* vkdev, cv::Mat& mat) {

	std::cout << "input data:" << std::endl;
	float* in_data = (float*)mat.data;
	for (int i = 0; i < 20; ++i) {
		std::cout << in_data[i] << " ";
	}
	std::cout << std::endl;

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
	vkmat.prepare_staging_buffer();
	vkmat.upload(mat);	//将cv::mat内容拷贝到vkmat.mapped_ptr()
	cmd.record_upload(vkmat);

	ret = relu.forward_inplace(vkmat, cmd);
	if (ret) {
		std::cout << "relu forward_inplace failed" << std::endl;
		return;
	}

	// download data	//必须放到submit_and_wait 之前
	vkmat.prepare_staging_buffer();
	cmd.record_download(vkmat);

	cmd.submit_and_wait();		//等待gpu执行完成
	cmd.reset();


	//vkmat.dowload();	//copy from VkMat::mapped_ptr() 

	std::cout << "out data:" << std::endl;
	float* out_data = (float*)vkmat.mapped_ptr();
	for (int i = 0; i < 20; ++i) {
		std::cout << out_data[i] << " ";
	}
	std::cout << std::endl;

	vkmat.discard_staging_buffer();
}

void resize_forward(VulkanDevice* vkdev, cv::Mat& mat) {

	Resize_vulkan resize;
	int ret = resize.create_pipeline(vkdev);
	if (ret) {
		std::cout << "create_pipeline err:" << ret << std::endl;
		return;
	}

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
	vkmat.prepare_staging_buffer();
	vkmat.upload(mat);	//将cv::mat内容拷贝到vkmat.mapped_ptr()
	cmd.record_upload(vkmat);

	cv::Mat outmat(240, 240, CV_32FC3);
	VkMat out_vkmat;
	out_vkmat.create_like(outmat, opt.blob_vkallocator, opt.staging_vkallocator);
	ret = resize.forward(vkmat, out_vkmat, cmd);
	if (ret) {
		std::cout << "relu forward_inplace failed" << std::endl;
		return;
	}

	// download data	//必须放到submit_and_wait 之前
	out_vkmat.prepare_staging_buffer();
	cmd.record_download(out_vkmat);

	cmd.submit_and_wait();		//等待gpu执行完成
	cmd.reset();


	out_vkmat.download(outmat);

	outmat.convertTo(outmat, CV_8UC3);

	cv::imshow("out", outmat);
	cv::waitKey(0);

	vkmat.discard_staging_buffer();
	out_vkmat.discard_staging_buffer();
}

void gpu_extract(VulkanDevice* vkdev, cv::Mat& mat) {
	std::cout << "start run vulkan" << std::endl;

	mat.convertTo(mat, CV_32FC3);

	//relu_forward(vkdev, mat);
	resize_forward(vkdev, mat);
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