#pragma once

#include <opencv2/opencv.hpp>
#include "../vulkan/vulkan_device.h"
#include "../vulkan/pipeline.h"
#include "../vulkan/command.h"
#include "../vulkan/vkmat.h"
#include "../vulkan/option.h"

namespace iml {
namespace train {

	class MatrixMulti
	{
	public:
		MatrixMulti();
		~MatrixMulti();

		void forward(std::vector<float>& left,
			std::vector<float>& right,
			std::vector<float>& out,
			int m, 
			int k,
			int n);
	};

}
} // namespace ncnn
