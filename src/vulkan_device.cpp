#include "vulkan_device.h"
#include <vulkan/vulkan.h>
#include <vector>

#define NCNN_MAX_GPU_COUNT 8

namespace iml {
namespace train {

	VulkanDevice* g_default_vkdev[NCNN_MAX_GPU_COUNT] = { 0 };
	extern int get_gpu_count();

	VulkanDevice::VulkanDevice(int device_index) {

	}

	VulkanDevice::~VulkanDevice() {

	}

	VulkanDevice* get_gpu_device(int device_index)
	{
		if (device_index < 0 || device_index >= get_gpu_count())
			return 0;

		if (!g_default_vkdev[device_index])
			g_default_vkdev[device_index] = new VulkanDevice(device_index);

		return g_default_vkdev[device_index];
	}
}
}