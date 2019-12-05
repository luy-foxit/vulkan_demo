#pragma once

#include <opencv2/opencv.hpp>
#include "../vulkan/vulkan_device.h"
#include "../vulkan/pipeline.h"
#include "../vulkan/command.h"
#include "../vulkan/vkmat.h"

namespace iml {
namespace train {

	class Convolution_vulkan
	{
	public:
		Convolution_vulkan();
		~Convolution_vulkan();

		int create_pipeline(const VulkanDevice* vkdev);
		int destroy_pipeline();

		int upload_model(VkTransfer& cmd, std::vector<float>& weight_data, std::vector<float>& bias_data);
		int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd) const;

	private:
		Pipeline* pipeline_convolution;
		VkMat weight_data_gpu;
		VkMat bias_data_gpu;
	};

}
} // namespace ncnn
