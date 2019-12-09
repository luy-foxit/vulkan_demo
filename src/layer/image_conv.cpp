#include "image_conv.h"


namespace iml {
namespace train {

	ImageConv::ImageConv()
	{
	}

	ImageConv::~ImageConv()
	{
	}

	int ImageConv::upload_model(
		std::vector<float>& weight_data,
		std::vector<float>& bias_data,
		int input_num,
		int output_num,
		int kernel_size) {

		_weight = weight_data;
		_bias = bias_data;

		_input_num = input_num;
		_output_num = output_num;
		_kernel_size = kernel_size;
		return 0;
	}

	static float kernel_multi(const float* in_ptr, const float* w_ptr, 
		int width, int height, 
		int kernel_size, 
		int out_x, int out_y) {
		float sum = 0.f;
		for (int y = 0; y < kernel_size; ++y) {
			for (int x = 0; x < kernel_size; ++x) {
				int in_y = out_y + (y - (kernel_size / 2));
				if (in_y < 0 || in_y >= height) {
					continue;
				}
				int in_x = out_x + (x - (kernel_size / 2));
				if (in_y < 0 || in_y >= width) {
					continue;
				}
				sum += in_ptr[in_y * width + in_x] * w_ptr[y * kernel_size + x];
			}
		}
		return sum;
	}

	int ImageConv::forward(std::vector<float>& bottom, std::vector<float>& top, int width, int height) const
	{
		top.resize(_output_num * width * height);

		int weight_count = _kernel_size * _kernel_size;
		
		for (int o = 0; o < _output_num; ++o) {

			const float* weight = &_weight[o * _input_num * weight_count];
			for (int h = 0; h < height; ++h) {
				for (int w = 0; w < width; ++w) {
					int out_idx = o * height * width + h * width + w;
					top[out_idx] = _bias[o];

					for (int i = 0; i < _input_num; ++i) {
						const float* in_ptr = &bottom[i * width * height];
						const float* w_ptr = weight + i * weight_count;
						top[out_idx] += kernel_multi(in_ptr, w_ptr, width, height, _kernel_size, w, h);
					}
				}
			}
		}

		return 0;
	}

}
} // namespace ncnn
