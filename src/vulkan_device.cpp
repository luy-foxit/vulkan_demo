#include "vulkan_device.h"
#include "gpu.h"
#include <vector>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include "allocator.h"

namespace iml {
namespace train {

	extern GpuInfo g_gpu_infos[NCNN_MAX_GPU_COUNT];

	VulkanDevice* g_default_vkdev[NCNN_MAX_GPU_COUNT] = { 0 };
	extern int get_gpu_count();

	struct layer_shader_registry_entry
	{
		const char* name;
		const uint32_t* spv_data;
		size_t spv_data_size;
	};
	#include "layer_shader_spv_data.h"
	static const layer_shader_registry_entry  layer_shader_registry[] =
	{
		#include "layer_shader_registry.h"
	};
	static const int layer_shader_registry_entry_count = sizeof(layer_shader_registry) / sizeof(layer_shader_registry_entry);

	static inline bool string_ends_with_fp16p(const char* name)
	{
		int len = strlen(name);
		if (len < 6)
			return false;

		return memcmp(name + len - 6, "_fp16p", 6) == 0;
	}

	static inline bool string_ends_with_fp16s(const char* name)
	{
		int len = strlen(name);
		if (len < 6)
			return false;

		return memcmp(name + len - 6, "_fp16s", 6) == 0;
	}

	static inline bool string_ends_with_fp16a(const char* name)
	{
		int len = strlen(name);
		if (len < 6)
			return false;

		return memcmp(name + len - 6, "_fp16a", 6) == 0;
	}

	VulkanDevice::VulkanDevice(int device_index) : info(g_gpu_infos[device_index]) {
		std::vector<const char*> enabledExtensions;
		if (info.support_VK_KHR_8bit_storage)
			enabledExtensions.push_back("VK_KHR_8bit_storage");
		if (info.support_VK_KHR_16bit_storage)
			enabledExtensions.push_back("VK_KHR_16bit_storage");
		if (info.support_VK_KHR_bind_memory2)
			enabledExtensions.push_back("VK_KHR_bind_memory2");
		if (info.support_VK_KHR_dedicated_allocation)
			enabledExtensions.push_back("VK_KHR_dedicated_allocation");
		if (info.support_VK_KHR_descriptor_update_template)
			enabledExtensions.push_back("VK_KHR_descriptor_update_template");
		if (info.support_VK_KHR_get_memory_requirements2)
			enabledExtensions.push_back("VK_KHR_get_memory_requirements2");
		if (info.support_VK_KHR_push_descriptor)
			enabledExtensions.push_back("VK_KHR_push_descriptor");
		if (info.support_VK_KHR_shader_float16_int8)
			enabledExtensions.push_back("VK_KHR_shader_float16_int8");
		if (info.support_VK_KHR_shader_float_controls)
			enabledExtensions.push_back("VK_KHR_shader_float_controls");
		if (info.support_VK_KHR_storage_buffer_storage_class)
			enabledExtensions.push_back("VK_KHR_storage_buffer_storage_class");

		void* enabledExtensionFeatures = 0;

		// enable int8 storage
		VkPhysicalDevice8BitStorageFeaturesKHR enabled8BitStorageFeatures;
		enabled8BitStorageFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR;
		enabled8BitStorageFeatures.pNext = 0;
		enabled8BitStorageFeatures.storageBuffer8BitAccess = info.support_int8_storage;
		enabled8BitStorageFeatures.uniformAndStorageBuffer8BitAccess = info.support_int8_storage;
		enabled8BitStorageFeatures.storagePushConstant8 = VK_FALSE;
		if (support_VK_KHR_get_physical_device_properties2 && info.support_VK_KHR_8bit_storage)
		{
			enabled8BitStorageFeatures.pNext = enabledExtensionFeatures;
			enabledExtensionFeatures = &enabled8BitStorageFeatures;
		}

		// enable fp16/int16 storage
		VkPhysicalDevice16BitStorageFeaturesKHR enabled16BitStorageFeatures;
		enabled16BitStorageFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR;
		enabled16BitStorageFeatures.pNext = 0;
		enabled16BitStorageFeatures.storageBuffer16BitAccess = info.support_fp16_storage;
		enabled16BitStorageFeatures.uniformAndStorageBuffer16BitAccess = info.support_fp16_storage;
		enabled16BitStorageFeatures.storagePushConstant16 = VK_FALSE;
		enabled16BitStorageFeatures.storageInputOutput16 = VK_FALSE;
		if (support_VK_KHR_get_physical_device_properties2 && info.support_VK_KHR_16bit_storage)
		{
			enabled16BitStorageFeatures.pNext = enabledExtensionFeatures;
			enabledExtensionFeatures = &enabled16BitStorageFeatures;
		}

		// enable fp16/int8 arithmetic
		VkPhysicalDeviceFloat16Int8FeaturesKHR enabledFloat16Int8Features;
		enabledFloat16Int8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR;
		enabledFloat16Int8Features.pNext = 0;
		enabledFloat16Int8Features.shaderFloat16 = info.support_fp16_arithmetic;
		enabledFloat16Int8Features.shaderInt8 = info.support_int8_arithmetic;
		if (support_VK_KHR_get_physical_device_properties2 && info.support_VK_KHR_shader_float16_int8)
		{
			enabledFloat16Int8Features.pNext = enabledExtensionFeatures;
			enabledExtensionFeatures = &enabledFloat16Int8Features;
		}

		std::vector<float> compute_queue_priorities(info.compute_queue_count, 1.f);// 0.f ~ 1.f
		std::vector<float> transfer_queue_priorities(info.transfer_queue_count, 1.f);// 0.f ~ 1.f

		VkDeviceQueueCreateInfo deviceQueueCreateInfos[2];
		deviceQueueCreateInfos[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		deviceQueueCreateInfos[0].pNext = 0;
		deviceQueueCreateInfos[0].flags = 0;
		deviceQueueCreateInfos[0].queueFamilyIndex = info.compute_queue_family_index;
		deviceQueueCreateInfos[0].queueCount = info.compute_queue_count;
		deviceQueueCreateInfos[0].pQueuePriorities = compute_queue_priorities.data();
		deviceQueueCreateInfos[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		deviceQueueCreateInfos[1].pNext = 0;
		deviceQueueCreateInfos[1].flags = 0;
		deviceQueueCreateInfos[1].queueFamilyIndex = info.transfer_queue_family_index;
		deviceQueueCreateInfos[1].queueCount = info.transfer_queue_count;
		deviceQueueCreateInfos[1].pQueuePriorities = transfer_queue_priorities.data();

		VkDeviceCreateInfo deviceCreateInfo;
		deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		deviceCreateInfo.pNext = enabledExtensionFeatures;
		deviceCreateInfo.flags = 0;
		if (info.compute_queue_family_index == info.transfer_queue_family_index)
		{
			deviceCreateInfo.queueCreateInfoCount = 1;
		}
		else
		{
			deviceCreateInfo.queueCreateInfoCount = 2;
		}
		deviceCreateInfo.pQueueCreateInfos = deviceQueueCreateInfos;
		deviceCreateInfo.enabledLayerCount = 0;
		deviceCreateInfo.ppEnabledLayerNames = 0;
		deviceCreateInfo.enabledExtensionCount = enabledExtensions.size();
		deviceCreateInfo.ppEnabledExtensionNames = enabledExtensions.data();
		deviceCreateInfo.pEnabledFeatures = 0;// VkPhysicalDeviceFeatures pointer

		VkResult ret = vkCreateDevice(info.physical_device, &deviceCreateInfo, 0, &device);
		if (ret != VK_SUCCESS)
		{
			fprintf(stderr, "vkCreateDevice failed %d\n", ret);
		}

		init_device_extension();

		create_shader_module();

		compute_queues.resize(info.compute_queue_count);
		blob_allocators.resize(info.compute_queue_count);
		staging_allocators.resize(info.compute_queue_count);
		for (uint32_t i = 0; i < info.compute_queue_count; i++)
		{
			vkGetDeviceQueue(device, info.compute_queue_family_index, i, &compute_queues[i]);
			blob_allocators[i] = new VkBlobBufferAllocator(this);
			staging_allocators[i] = new VkStagingBufferAllocator(this);
		}
		if (info.compute_queue_family_index != info.transfer_queue_family_index)
		{
			transfer_queues.resize(info.transfer_queue_count);
			for (uint32_t i = 0; i < info.transfer_queue_count; i++)
			{
				vkGetDeviceQueue(device, info.transfer_queue_family_index, i, &transfer_queues[i]);
			}
		}
	}

	VulkanDevice::~VulkanDevice() {

	}

	VkShaderModule VulkanDevice::get_shader_module(const char* name) const
	{
		for (int i = 0; i < layer_shader_registry_entry_count; i++)
		{
			if (strcmp(layer_shader_registry[i].name, name) == 0)
				return shader_modules[i];
		}

		fprintf(stderr, "no such shader module %s\n", name);
		return 0;
	}

	int VulkanDevice::init_device_extension()
	{
		if (info.support_VK_KHR_descriptor_update_template)
		{
			vkCreateDescriptorUpdateTemplateKHR = (PFN_vkCreateDescriptorUpdateTemplateKHR)vkGetDeviceProcAddr(device, "vkCreateDescriptorUpdateTemplateKHR");
			vkDestroyDescriptorUpdateTemplateKHR = (PFN_vkDestroyDescriptorUpdateTemplateKHR)vkGetDeviceProcAddr(device, "vkDestroyDescriptorUpdateTemplateKHR");
			vkUpdateDescriptorSetWithTemplateKHR = (PFN_vkUpdateDescriptorSetWithTemplateKHR)vkGetDeviceProcAddr(device, "vkUpdateDescriptorSetWithTemplateKHR");
		}

		if (info.support_VK_KHR_get_memory_requirements2)
		{
			vkGetImageMemoryRequirements2KHR = (PFN_vkGetImageMemoryRequirements2KHR)vkGetDeviceProcAddr(device, "vkGetImageMemoryRequirements2KHR");
			vkGetBufferMemoryRequirements2KHR = (PFN_vkGetBufferMemoryRequirements2KHR)vkGetDeviceProcAddr(device, "vkGetBufferMemoryRequirements2KHR");
			vkGetImageSparseMemoryRequirements2KHR = (PFN_vkGetImageSparseMemoryRequirements2KHR)vkGetDeviceProcAddr(device, "vkGetImageSparseMemoryRequirements2KHR");
		}

		if (info.support_VK_KHR_push_descriptor)
		{
			if (info.support_VK_KHR_descriptor_update_template)
			{
				vkCmdPushDescriptorSetWithTemplateKHR = (PFN_vkCmdPushDescriptorSetWithTemplateKHR)vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetWithTemplateKHR");
			}

			vkCmdPushDescriptorSetKHR = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR");
		}

		return 0;
	}

	VkShaderModule VulkanDevice::compile_shader_module(const uint32_t* spv_data, size_t spv_data_size) const
	{
		VkShaderModuleCreateInfo shaderModuleCreateInfo;
		shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		shaderModuleCreateInfo.pNext = 0;
		shaderModuleCreateInfo.flags = 0;
		shaderModuleCreateInfo.codeSize = spv_data_size;
		shaderModuleCreateInfo.pCode = spv_data;

		VkShaderModule shader_module;
		VkResult ret = vkCreateShaderModule(device, &shaderModuleCreateInfo, 0, &shader_module);
		if (ret != VK_SUCCESS)
		{
			fprintf(stderr, "vkCreateShaderModule failed %d\n", ret);
			return 0;
		}

		return shader_module;
	}

	int VulkanDevice::create_shader_module()
	{
		shader_modules.resize(layer_shader_registry_entry_count, VK_NULL_HANDLE);

		for (int i = 0; i < layer_shader_registry_entry_count; i++)
		{
			const char* shader_name = layer_shader_registry[i].name;

			if (!info.support_fp16_packed)
			{
				if (string_ends_with_fp16p(shader_name))
					continue;
			}

			if (!info.support_fp16_storage)
			{
				if (string_ends_with_fp16s(shader_name))
					continue;
			}

			if (!info.support_fp16_arithmetic)
			{
				if (string_ends_with_fp16a(shader_name))
					continue;
			}

			VkShaderModule shader_module = compile_shader_module(layer_shader_registry[i].spv_data, layer_shader_registry[i].spv_data_size);
			if (shader_module == 0)
			{
				fprintf(stderr, "compile_shader_module %s failed\n", shader_name);
				return -1;
			}

			shader_modules[i] = shader_module;

			//         fprintf(stderr, "shader_module %s created\n", shader_name);
		}

		return 0;
	}

	void VulkanDevice::destroy_shader_module()
	{
		for (int i = 0; i < (int)shader_modules.size(); i++)
		{
			vkDestroyShaderModule(device, shader_modules[i], 0);
		}

		shader_modules.clear();
	}


	VkAllocator* VulkanDevice::acquire_blob_allocator() const
	{
		for (int i = 0; i < (int)blob_allocators.size(); i++)
		{
			VkAllocator* allocator = blob_allocators[i];
			if (allocator)
			{
				blob_allocators[i] = 0;
				return allocator;
			}
		}

		// out of blob allocator
		return 0;
	}

	void VulkanDevice::reclaim_blob_allocator(VkAllocator* allocator) const
	{
		for (int i = 0; i < (int)blob_allocators.size(); i++)
		{
			if (!blob_allocators[i])
			{
				blob_allocators[i] = allocator;
				return;
			}
		}

		fprintf(stderr, "FATAL ERROR! reclaim_blob_allocator get wild allocator %p\n", allocator);
	}

	VkAllocator* VulkanDevice::acquire_staging_allocator() const
	{
		for (int i = 0; i < (int)staging_allocators.size(); i++)
		{
			VkAllocator* allocator = staging_allocators[i];
			if (allocator)
			{
				staging_allocators[i] = 0;
				return allocator;
			}
		}

		// out of staging allocator
		return 0;
	}

	void VulkanDevice::reclaim_staging_allocator(VkAllocator* allocator) const
	{
		for (int i = 0; i < (int)staging_allocators.size(); i++)
		{
			if (!staging_allocators[i])
			{
				staging_allocators[i] = allocator;
				return;
			}
		}

		fprintf(stderr, "FATAL ERROR! reclaim_staging_allocator get wild allocator %p\n", allocator);
	}


	VkQueue VulkanDevice::acquire_queue(uint32_t queue_family_index) const
	{
		if (queue_family_index != info.compute_queue_family_index && queue_family_index != info.transfer_queue_family_index)
		{
			fprintf(stderr, "invalid queue_family_index %u\n", queue_family_index);
			return 0;
		}

		std::vector<VkQueue>& queues = queue_family_index == info.compute_queue_family_index ? compute_queues : transfer_queues;
		for (int i = 0; i < (int)queues.size(); i++)
		{
			VkQueue queue = queues[i];
			if (queue)
			{
				queues[i] = 0;
				return queue;
			}
		}

		// out of hardware queue
		return 0;
	}

	void VulkanDevice::reclaim_queue(uint32_t queue_family_index, VkQueue queue) const
	{
		if (queue_family_index != info.compute_queue_family_index && queue_family_index != info.transfer_queue_family_index)
		{
			fprintf(stderr, "invalid queue_family_index %u\n", queue_family_index);
			return;
		}

		std::vector<VkQueue>& queues = queue_family_index == info.compute_queue_family_index ? compute_queues : transfer_queues;
		for (int i = 0; i < (int)queues.size(); i++)
		{
			if (!queues[i])
			{
				queues[i] = queue;
				return;
			}
		}

		fprintf(stderr, "FATAL ERROR! reclaim_queue get wild queue %u %p\n", queue_family_index, queue);
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
