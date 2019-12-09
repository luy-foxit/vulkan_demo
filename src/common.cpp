#include "common.h"
#include <time.h>
#include <random>


namespace iml {
	namespace train {

		void random_vector(std::vector<float>& matrix) {
			int size = static_cast<int>(matrix.size());
			unsigned int min = 0;
			unsigned int max = 255;
			clock_t start = clock();
			for (int i = 0; i < size; ++i) {
				unsigned int seed = static_cast<unsigned int>(i + start);
				static std::default_random_engine e(seed);
				static std::uniform_real_distribution<double> u(min, max);
				matrix[i] = static_cast<float>(static_cast<int>(u(e))) / 255.0f;
			}
		}

		void random_weight(int input_num, int output_num, int kernel_size, std::vector<float>& weight) {
			int size = kernel_size * kernel_size * input_num * output_num;
			weight.resize(size);
			random_vector(weight);
		}

		void random_bias(int output_num, std::vector<float>& bias) {
			bias.resize(output_num);
			random_vector(bias);
		}

		void clear_vector(std::vector<float>& v, float num/* = 0.f*/) {
			for (auto& f : v) {
				f = num;
			}
		}

		void check_result(std::vector<float>& l, std::vector<float>& r) {
			for (int i = 0; i < l.size(); ++i) {
				if (l[i] != r[i]) {
					std::cout << "compare error in idx:" << i << "." << l[i] << ":" << r[i] << std::endl;
					if (i > 100) {
						return;
					}
				}
			}
		}

	}
}