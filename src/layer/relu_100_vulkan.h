#pragma once

#include <opencv2/opencv.hpp>
#include "../vulkan_device.h"
#include "../pipeline.h"
#include "../command.h"

namespace iml {
namespace train {

	class ReLU_vulkan
	{
	public:
		ReLU_vulkan();
		~ReLU_vulkan();

		int create_pipeline(const VulkanDevice* vkdev);
		int destroy_pipeline();

		int forward_inplace(cv::Mat& bottom_top_blob, VkCompute& cmd) const;

	private:
		Pipeline* pipeline_relu;
	};

}
} // namespace ncnn


