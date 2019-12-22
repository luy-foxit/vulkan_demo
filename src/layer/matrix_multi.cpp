#include "matrix_multi.h"


namespace iml {
namespace train {

	static void AddDot(int k, float *x, float *y, int n, float *gamma);

	void MMultBase(float* A, float* B, float* C, int m, int n, int k) {
		//A: m*k; B: k*n; C: m*n
		//普通做法, 行乘列
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				AddDot(k, &A[i*k], &B[j], n, &C[i*n + j]);
			}
		}
	}

	static void AddDot(int k, float *x, float *y, int n, float *gamma)
	{
		for (int p = 0; p < k; p++) {
			*gamma += x[p] * y[p * n];
		}
	}

	MatrixMulti::MatrixMulti()
	{
	}

	MatrixMulti::~MatrixMulti()
	{
	}

	void MatrixMulti::forward(std::vector<float>& left,
		std::vector<float>& right,
		std::vector<float>& out,
		int m,
		int n,
		int k) {
		out.resize(m*n);
		MMultBase(&left[0], &right[0], &out[0], m, n ,k);
	}

}
} // namespace ncnn
