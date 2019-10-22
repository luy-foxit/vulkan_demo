#pragma once

#include <opencv2/opencv.hpp>
#include "../vulkan_device.h"
#include "../pipeline.h"
#include "../command.h"
#include "../vkmat.h"

namespace iml {
namespace train {

	class Divide_vulkan
	{
	public:
		Divide_vulkan(float div_num);
		~Divide_vulkan();

		int create_pipeline(const VulkanDevice* vkdev);
		int destroy_pipeline();

		int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd) const;

	private:
		Pipeline* pipeline_relu;
        float _div_num;
	};

}
} // namespace ncnn


