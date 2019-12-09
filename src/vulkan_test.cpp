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

void divide_forward(VulkanDevice* vkdev, Option& opt, cv::Mat& mat) {

	std::cout << "input data:" << std::endl;
	float* in_data = (float*)mat.data;
	for (int i = 0; i < 20; ++i) {
		std::cout << in_data[i] << " ";
	}
	std::cout << std::endl;

	Divide_vulkan div(2.5f);
	int ret = div.create_pipeline(vkdev);
	if (ret) {
		std::cout << "create_pipeline err:" << ret << std::endl;
		return;
	}

	VkCompute cmd(vkdev);
	VkMat vkmat;
	vkmat.create_like(mat, opt.blob_vkallocator, opt.staging_vkallocator);
	vkmat.prepare_staging_buffer();
	vkmat.upload(mat);	//将cv::mat内容拷贝到vkmat.mapped_ptr()
	cmd.record_upload(vkmat);

	ret = div.forward_inplace(vkmat, cmd);
	if (ret) {
		std::cout << "div forward_inplace failed" << std::endl;
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

void resize_forward(VulkanDevice* vkdev, Option& opt, cv::Mat& mat) {

	Resize_vulkan resize;
	int ret = resize.create_pipeline(vkdev);
	if (ret) {
		std::cout << "create_pipeline err:" << ret << std::endl;
		return;
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
		std::cout << "divide forward_inplace failed" << std::endl;
		return;
	}

	cmd.submit_and_wait();		//等待gpu执行完成
	cmd.reset();		//重置并且重新开始command

	VkCompute cmd2(vkdev);
	out_vkmat.prepare_staging_buffer();
	cmd2.record_download(out_vkmat);

	cmd2.submit_and_wait();
	cmd2.reset();

	out_vkmat.download(outmat);

	outmat.convertTo(outmat, CV_8UC3);

	cv::imshow("out", outmat);
	cv::waitKey(0);

	vkmat.discard_staging_buffer();
	out_vkmat.discard_staging_buffer();
}

void upload_weights(VulkanDevice* vkdev, Convolution_vulkan& conv) {
	VkTransfer cmd(vkdev);

	// create gpu device allocator if null
	VkAllocator* weight_vkallocator = new VkWeightBufferAllocator(vkdev);
	VkAllocator* weight_staging_vkallocator = new VkWeightStagingBufferAllocator(vkdev);

	cmd.weight_vkallocator = weight_vkallocator;
	cmd.staging_vkallocator = weight_staging_vkallocator;
	
	std::vector<float> weight = {
		0.1f, 0.1f, 0.1f,
		0.1f, 0.2f, 0.1f,
		0.1f, 0.1f, 0.1f,
	};
	std::vector<float> bias = { 10.0f, 10.0f, 10.0f };
	conv.upload_model(cmd, weight, bias);			//upload weight

	cmd.submit_and_wait();
}

void convolution_forward(VulkanDevice* vkdev, Option& opt, cv::Mat& mat) {

	Convolution_vulkan conv;
	int ret = conv.create_pipeline(vkdev);
	if (ret) {
		std::cout << "create_pipeline err:" << ret << std::endl;
		return;
	}

	upload_weights(vkdev, conv);

	VkCompute cmd(vkdev);
	VkMat vkmat;
	vkmat.create_like(mat, opt.blob_vkallocator, opt.staging_vkallocator);
	vkmat.prepare_staging_buffer();
	vkmat.upload(mat);	//将cv::mat内容拷贝到vkmat.mapped_ptr()
	cmd.record_upload(vkmat);

	ret = conv.forward_inplace(vkmat, cmd);
	if (ret) {
		std::cout << "conv forward_inplace failed" << std::endl;
		return;
	}

	// download data	//必须放到submit_and_wait 之前
	vkmat.prepare_staging_buffer();
	cmd.record_download(vkmat);

	cmd.submit_and_wait();		//等待gpu执行完成
	cmd.reset();


	cv::Mat outmat = mat.clone();
	vkmat.download(outmat);

	outmat.convertTo(outmat, CV_8UC3);

	cv::imshow("out", outmat);
	cv::waitKey(0);

	vkmat.discard_staging_buffer();
}
