#pragma once

namespace iml {
namespace train {

	int get_default_gpu_index();

	class VulkanDevice {
	public:
		VulkanDevice(int device_index = get_default_gpu_index());
		~VulkanDevice();
	};

	extern VulkanDevice* g_default_vkdev[];

}
}