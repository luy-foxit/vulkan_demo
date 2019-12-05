#pragma once

#include <opencv2/opencv.hpp>
#include "../vulkan/vulkan_device.h"
#include "../vulkan/pipeline.h"
#include "../vulkan/command.h"
#include "../vulkan/vkmat.h"

namespace iml {
namespace train {

	class Resize_vulkan
	{
	public:
		Resize_vulkan();
		~Resize_vulkan();

		int create_pipeline(const VulkanDevice* vkdev);
		int destroy_pipeline();

		int forward(VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd) const;

	private:
		Pipeline* pipeline_resize;
	};

}
} // namespace ncnn
