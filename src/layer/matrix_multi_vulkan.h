#pragma once

#include <opencv2/opencv.hpp>
#include "../vulkan/vulkan_device.h"
#include "../vulkan/pipeline.h"
#include "../vulkan/command.h"
#include "../vulkan/vkmat.h"
#include "../vulkan/option.h"

namespace iml {
namespace train {

	class MatrixMulti_vulkan
	{
	public:
		MatrixMulti_vulkan();
		~MatrixMulti_vulkan();

		int create_pipeline(const VulkanDevice* vkdev);
		void destroy_pipeline();

		void forward(
			VkMat& left_blob,
			VkMat& right_blob,
			VkMat& top_blob,
			VkCompute& cmd,
			Option& opt,
			int m, 
			int k,
			int n);

	private:
		Pipeline* pipeline_mm;
	};

}
} // namespace ncnn
