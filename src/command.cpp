#include "command.h"

#include <stdio.h>

namespace iml {
namespace train {

	Command::Command(const VulkanDevice* _vkdev, uint32_t _queue_family_index) : vkdev(_vkdev), queue_family_index(_queue_family_index)
	{
	}

	Command::~Command()
	{
	}

	VkCompute::VkCompute(const VulkanDevice* _vkdev) : Command(_vkdev, _vkdev->info.compute_queue_family_index)
	{
	}

	VkCompute::~VkCompute()
	{
	}

	VkTransfer::VkTransfer(const VulkanDevice* _vkdev) : Command(_vkdev, _vkdev->info.transfer_queue_family_index)
	{
		
	}

	VkTransfer::~VkTransfer()
	{
	}

}
} // namespace ncnn

