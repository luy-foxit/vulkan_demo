#include <iostream>
#include <opencv2/opencv.hpp>
#include "vulkan/gpu.h"
#include "vulkan/vkmat.h"
#include "vulkan/option.h"
#include "layer/matrix_multi.h"
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

void matrix_multi_test(VulkanDevice* vkdev, Option& opt) {

	int size = 600;

	int m = size;
	int n = size;
	int k = size;

	std::vector<float> left(m*k);
	std::vector<float> right(k*n);
	random_vector(left);
	random_vector(right);
	std::vector<float> output1;

	matrix_multi_forward(left, right, output1, m, n, k);
	for (int i = 0; i < 10; ++i) {
		std::cout << output1[i] << " ";
	}
	std::cout << std::endl;
}