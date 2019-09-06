#pragma once

#include <opencv2/opencv.hpp>
#include "../vulkan_device.h"
#include "../pipeline.h"
#include "../command.h"
#include "../vkmat.h"

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
