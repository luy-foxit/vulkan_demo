#include <iostream>
#include <opencv2/opencv.hpp>
#include "vulkan/gpu.h"
#include "vulkan/vkmat.h"
#include "vulkan/option.h"
#include "layer/matrix_multi.h"
#include "layer/matrix_multi_vulkan.h"
#include "common.h"

using namespace iml::train;

void matrix_multi_forward(std::vector<float>& left,
	std::vector<float>& right,
	std::vector<float>& out, 
	int m, 
	int n, 
	int k) {
	MatrixMulti mm_matrix;
	mm_matrix.forward(left, right, out, m, n, k);
}

void matrix_multi_vulkan_forward(
	VulkanDevice* vkdev,
	Option& opt,
	std::vector<float>& left,
	std::vector<float>& right,
	std::vector<float>& out,
	int m,
	int n,
	int k) {

	MatrixMulti_vulkan mm_matrix;

	int ret = mm_matrix.create_pipeline(vkdev);
	if (ret) {
		std::cout << "create_pipeline err:" << ret << std::endl;
		return;
	}

	int elemsize = sizeof(float);
	int elempack = 1;

	VkCompute cmd(vkdev);
	VkMat left_blob;
	left_blob.create(k, m, 1, elemsize, elempack, opt.blob_vkallocator, opt.staging_vkallocator);
	left_blob.prepare_staging_buffer();
	left_blob.upload(left);	//将vector内容拷贝到left_blob.mapped_ptr()
	cmd.record_upload(left_blob);

	VkMat right_blob;
	right_blob.create(n, k, 1, elemsize, elempack, opt.blob_vkallocator, opt.staging_vkallocator);
	right_blob.prepare_staging_buffer();
	right_blob.upload(right);	//将vector内容拷贝到right_blob.mapped_ptr()
	cmd.record_upload(right_blob);

	VkMat top_blob;
	mm_matrix.forward(left_blob, right_blob, top_blob, cmd, opt, m, n, k);

	top_blob.prepare_staging_buffer();
	cmd.record_download(top_blob);

	cmd.submit_and_wait();		//等待gpu执行完成
	cmd.reset();

	top_blob.download(out);

	left_blob.discard_staging_buffer();
	right_blob.discard_staging_buffer();
}

void matrix_multi_test(VulkanDevice* vkdev, Option& opt) {

	int size = 600;

	int m = size;
	int n = size;
	int k = size;

	std::vector<float> left(m*k);
	std::vector<float> right(k*n);
	random_vector(left);
	random_vector(right);
	std::vector<float> output_cpu;

	matrix_multi_forward(left, right, output_cpu, m, n, k);
	for (int i = 0; i < 10; ++i) {
		std::cout << output_cpu[i] << " ";
	}
	std::cout << std::endl;

	std::vector<float> output_vulkan1;
	matrix_multi_vulkan_forward(vkdev, opt, left, right, output_vulkan1, m, n, k);
	for (int i = 0; i < 10; ++i) {
		std::cout << output_vulkan1[i] << " ";
	}
	std::cout << std::endl;
}