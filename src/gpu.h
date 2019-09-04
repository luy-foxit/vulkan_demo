#pragma once

#include <vulkan/vulkan.h>

namespace iml {
namespace train {

	extern int support_VK_KHR_get_physical_device_properties2;
	extern int support_VK_EXT_debug_utils;

	// VK_KHR_get_physical_device_properties2
	extern PFN_vkGetPhysicalDeviceFeatures2KHR vkGetPhysicalDeviceFeatures2KHR;
	extern PFN_vkGetPhysicalDeviceProperties2KHR vkGetPhysicalDeviceProperties2KHR;
	extern PFN_vkGetPhysicalDeviceFormatProperties2KHR vkGetPhysicalDeviceFormatProperties2KHR;
	extern PFN_vkGetPhysicalDeviceImageFormatProperties2KHR vkGetPhysicalDeviceImageFormatProperties2KHR;
	extern PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR vkGetPhysicalDeviceQueueFamilyProperties2KHR;
	extern PFN_vkGetPhysicalDeviceMemoryProperties2KHR vkGetPhysicalDeviceMemoryProperties2KHR;
	extern PFN_vkGetPhysicalDeviceSparseImageFormatProperties2KHR vkGetPhysicalDeviceSparseImageFormatProperties2KHR;

	int create_gpu_instance();
	void destroy_gpu_instance();

	struct GpuInfo {
		VkPhysicalDevice physical_device;

		// info
		uint32_t api_version;
		uint32_t driver_version;
		uint32_t vendor_id;
		uint32_t device_id;
		uint8_t pipeline_cache_uuid[VK_UUID_SIZE];

		// 0 = ∂¿¡¢œ‘ø®
		// 1 = ºØ≥…œ‘ø®
		// 2 = –Èƒ‚œ‘ø®
		// 3 = cpu
		int type;

		// hardware capability
		uint32_t max_shared_memory_size;
		uint32_t max_workgroup_count[3];
		uint32_t max_workgroup_invocations;
		uint32_t max_workgroup_size[3];
		size_t memory_map_alignment;
		size_t buffer_offset_alignment;
		float timestamp_period;

		// runtime
		uint32_t compute_queue_family_index;
		uint32_t transfer_queue_family_index;

		uint32_t compute_queue_count;
		uint32_t transfer_queue_count;

		uint32_t unified_memory_index;
		uint32_t device_local_memory_index;
		uint32_t host_visible_memory_index;

		// fp16 and int8 feature
		bool support_fp16_packed;
		bool support_fp16_storage;
		bool support_fp16_arithmetic;
		bool support_int8_storage;
		bool support_int8_arithmetic;

		// extension capability
		int support_VK_KHR_8bit_storage;
		int support_VK_KHR_16bit_storage;
		int support_VK_KHR_bind_memory2;
		int support_VK_KHR_dedicated_allocation;
		int support_VK_KHR_descriptor_update_template;
		int support_VK_KHR_get_memory_requirements2;
		int support_VK_KHR_push_descriptor;
		int support_VK_KHR_shader_float16_int8;
		int support_VK_KHR_shader_float_controls;
		int support_VK_KHR_storage_buffer_storage_class;
	};

	int get_default_gpu_index();
	int get_gpu_count();

}
}