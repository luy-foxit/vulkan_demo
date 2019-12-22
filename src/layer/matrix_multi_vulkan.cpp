#include "matrix_multi_vulkan.h"


namespace iml {
namespace train {

	static void AddDot(int k, float *x, float *y, int n, float *gamma);

	MatrixMulti_vulkan::MatrixMulti_vulkan()
	{
	}

	MatrixMulti_vulkan::~MatrixMulti_vulkan()
	{
	}

	void MatrixMulti_vulkan::forward(std::vector<float>& left,
		std::vector<float>& right,
		std::vector<float>& out,
		int m,
		int n,
		int k) {
		out.resize(m*n);
		//MMultBase(&left[0], &right[0], &out[0], m, n ,k);
	}

}
} // namespace ncnn
