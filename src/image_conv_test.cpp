#include <iostream>
#include <opencv2/opencv.hpp>
#include "vulkan/gpu.h"
#include "vulkan/vkmat.h"
#include "vulkan/option.h"
#include "layer/image_conv_vulkan.h"
#include "layer/image_conv.h"
#include "common.h"

using namespace iml::train;

void upload_image_conv_weights(
	VulkanDevice* vkdev, 
	ImageConv_vulkan& conv, 
	std::vector<float>& weight, 
	std::vector<float>& bias) 
{
	VkTransfer cmd(vkdev);

	// create gpu device allocator if null
	VkAllocator* weight_vkallocator = new VkWeightBufferAllocator(vkdev);
	VkAllocator* weight_staging_vkallocator = new VkWeightStagingBufferAllocator(vkdev);

	cmd.weight_vkallocator = weight_vkallocator;
	cmd.staging_vkallocator = weight_staging_vkallocator;

	conv.upload_model(cmd, weight, bias);			//upload weight

	cmd.submit_and_wait();
}

void image_conv_vulkan_forward(
	VulkanDevice* vkdev, 
	Option& opt, 
	cv::Mat& mat,
	std::vector<float>& output,
	std::vector<float>& weight,
	std::vector<float>& bias,
	int input_num,
	int output_num,
	int kernel_size)
{
	ImageConv_vulkan conv;
	int ret = conv.create_pipeline(vkdev, kernel_size);
	if (ret) {
		std::cout << "create_pipeline err:" << ret << std::endl;
		return;
	}

	upload_image_conv_weights(vkdev, conv, weight, bias);

	VkCompute cmd(vkdev);
	VkMat vkmat;
	vkmat.create_like(mat, opt.blob_vkallocator, opt.staging_vkallocator);
	vkmat.prepare_staging_buffer();
	vkmat.upload(mat);	//将cv::mat内容拷贝到vkmat.mapped_ptr()
	cmd.record_upload(vkmat);

	VkMat vkout;
	ret = conv.forward(vkmat, vkout, cmd, opt, output_num);
	if (ret) {
		std::cout << "conv forward_inplace failed" << std::endl;
		return;
	}

	vkout.prepare_staging_buffer();
	cmd.record_download(vkout);

	cmd.submit_and_wait();		//等待gpu执行完成
	cmd.reset();

	vkout.download(output);

	vkmat.discard_staging_buffer();
}

static void mat_to_vector(cv::Mat& mat, std::vector<float>& vec) {
	cv::Mat in_mat;
	mat.convertTo(in_mat, CV_32FC3);
	int vec_size = in_mat.rows * in_mat.cols * in_mat.channels();
	vec.resize(vec_size);

	for (int i = 0; i < vec_size; ++i) {
		memcpy(&vec[0], in_mat.data, vec_size * sizeof(float));
	}
}

void image_conv_forward(cv::Mat& mat,
	std::vector<float>& output,
	std::vector<float>& weight, 
	std::vector<float>& bias, 
	int input_num, 
	int output_num, 
	int kernel_size) {

	ImageConv conv;
	
	conv.upload_model(weight, bias, input_num, output_num, kernel_size);
	std::vector<float> bottom;
	mat_to_vector(mat, bottom);
	int ret = conv.forward(bottom, output, mat.cols, mat.rows);
	if (ret) {
		std::cout << "conv forward_inplace failed" << std::endl;
		return;
	}
}

void image_conv_test(VulkanDevice* vkdev, Option& opt, cv::Mat& mat) {
	cv::Mat in_mat;
	cv::resize(mat, in_mat, cv::Size(256, 256));

	int kernel_size = 3;
	int input_num = 3;
	int output_num = 4;

	std::vector<float> weight, bias;
	random_weight(input_num, output_num, kernel_size, weight);
	random_bias(output_num, bias);
	std::vector<float> output;

	//cpu test
	image_conv_forward(in_mat, output, weight, bias, input_num, output_num, kernel_size);

	image_conv_vulkan_forward(vkdev, opt, in_mat, output, weight, bias, input_num, output_num, kernel_size);
}