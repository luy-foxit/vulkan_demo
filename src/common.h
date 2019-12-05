#pragma once

#include <iostream>
#include <vector>

namespace iml {
namespace train {

	void random_vector(std::vector<float>& matrix);
	void random_weight(int input_num, int output_num, int kernel_size, std::vector<float>& weight);
	void random_bias(int output_num, std::vector<float>& bias);
	
	void clear_vector(std::vector<float>& v, float num = 0.f);
	void check_result(std::vector<float>& l, std::vector<float>& r);

}
}