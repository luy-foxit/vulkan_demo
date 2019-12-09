#pragma once

#include <opencv2/opencv.hpp>
#include "../vulkan/vulkan_device.h"
#include "../vulkan/pipeline.h"
#include "../vulkan/command.h"
#include "../vulkan/vkmat.h"
#include "../vulkan/option.h"

namespace iml {
namespace train {

	class ImageConv
	{
	public:
		ImageConv();
		~ImageConv();

		int upload_model(std::vector<float>& weight_data, std::vector<float>& bias_data, int input_num, int output_num, int kernel_size);
		int forward(std::vector<float>& bottom_blob, 
			std::vector<float>& top_blob,
			int w, 
			int h) const;

	private:
		int _input_num;
		int _output_num;
		int _kernel_size;

		std::vector<float> _weight;
		std::vector<float> _bias;
	};

}
} // namespace ncnn
