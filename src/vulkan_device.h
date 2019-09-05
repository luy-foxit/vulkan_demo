#pragma once

#include <vulkan/vulkan.h>
#include <vector>

#define NCNN_MAX_GPU_COUNT 8

namespace iml {
namespace train {

	int get_default_gpu_index();
	struct GpuInfo;
	class VkAllocator;
	class VulkanDevice {
	public:
		VulkanDevice(int device_index = get_default_gpu_index());
		~VulkanDevice();

		VkShaderModule get_shader_module(const char* name) const;
		VkShaderModule compile_shader_module(const uint32_t* spv_data, size_t spv_data_size) const;

		// allocator on this device
		VkAllocator* acquire_blob_allocator() const;
		void reclaim_blob_allocator(VkAllocator* allocator) const;

		VkAllocator* acquire_staging_allocator() const;
		void reclaim_staging_allocator(VkAllocator* allocator) const;


	protected:
		int init_device_extension();
		int create_shader_module();
		void destroy_shader_module();

	public:
		GpuInfo& info;
		VkDevice vkdevice() const { return device; }

		// VK_KHR_descriptor_update_template
		PFN_vkCreateDescriptorUpdateTemplateKHR vkCreateDescriptorUpdateTemplateKHR;
		PFN_vkDestroyDescriptorUpdateTemplateKHR vkDestroyDescriptorUpdateTemplateKHR;
		PFN_vkUpdateDescriptorSetWithTemplateKHR vkUpdateDescriptorSetWithTemplateKHR;

		// VK_KHR_get_memory_requirements2
		PFN_vkGetImageMemoryRequirements2KHR vkGetImageMemoryRequirements2KHR;
		PFN_vkGetBufferMemoryRequirements2KHR vkGetBufferMemoryRequirements2KHR;
		PFN_vkGetImageSparseMemoryRequirements2KHR vkGetImageSparseMemoryRequirements2KHR;

		// VK_KHR_push_descriptor
		PFN_vkCmdPushDescriptorSetWithTemplateKHR vkCmdPushDescriptorSetWithTemplateKHR;
		PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR;

	private:
		VkDevice device;
		std::vector<VkShaderModule> shader_modules;

		// hardware queue
		mutable std::vector<VkQueue> compute_queues;	//运算queue
		mutable std::vector<VkQueue> transfer_queues;	//数据传输queue

		// default blob allocator for each queue
		mutable std::vector<VkAllocator*> blob_allocators;

		// default staging allocator for each queue
		mutable std::vector<VkAllocator*> staging_allocators;

	};

	VulkanDevice* get_gpu_device(int device_index = get_default_gpu_index());

}
}