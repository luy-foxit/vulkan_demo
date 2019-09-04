#include "gpu.h"
#include <vector>
#include "vulkan_device.h"

#define ENABLE_VALIDATION_LAYER 0

namespace iml {
namespace train {

	int support_VK_KHR_get_physical_device_properties2 = 0;
	int support_VK_EXT_debug_utils = 0;

	static VkInstance g_instance = 0;
	static int g_gpu_count = 0;
	static int g_default_gpu_index = -1;

	GpuInfo g_gpu_infos[NCNN_MAX_GPU_COUNT];
	extern VulkanDevice* g_default_vkdev[];

	// VK_KHR_get_physical_device_properties2
	PFN_vkGetPhysicalDeviceFeatures2KHR vkGetPhysicalDeviceFeatures2KHR = 0;
	PFN_vkGetPhysicalDeviceProperties2KHR vkGetPhysicalDeviceProperties2KHR = 0;
	PFN_vkGetPhysicalDeviceFormatProperties2KHR vkGetPhysicalDeviceFormatProperties2KHR = 0;
	PFN_vkGetPhysicalDeviceImageFormatProperties2KHR vkGetPhysicalDeviceImageFormatProperties2KHR = 0;
	PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR vkGetPhysicalDeviceQueueFamilyProperties2KHR = 0;
	PFN_vkGetPhysicalDeviceMemoryProperties2KHR vkGetPhysicalDeviceMemoryProperties2KHR = 0;
	PFN_vkGetPhysicalDeviceSparseImageFormatProperties2KHR vkGetPhysicalDeviceSparseImageFormatProperties2KHR = 0;

	static int create_instance() {
		VkResult ret;

		std::vector<const char*> enabledLayers;

#if ENABLE_VALIDATION_LAYER
		uint32_t instanceLayerPropertyCount;
		ret = vkEnumerateInstanceLayerProperties(&instanceLayerPropertyCount, NULL);
		if (ret != VK_SUCCESS)
		{
			fprintf(stderr, "vkEnumerateInstanceLayerProperties failed %d\n", ret);
			return -1;
		}

		std::vector<VkLayerProperties> instanceLayerProperties(instanceLayerPropertyCount);
		ret = vkEnumerateInstanceLayerProperties(&instanceLayerPropertyCount, instanceLayerProperties.data());
		if (ret != VK_SUCCESS)
		{
			fprintf(stderr, "vkEnumerateInstanceLayerProperties failed %d\n", ret);
			return -1;
		}

		for (uint32_t i = 0; i < instanceLayerPropertyCount; i++)
		{
			const VkLayerProperties& lp = instanceLayerProperties[i];
			//         fprintf(stderr, "instance layer %s = %u\n", lp.layerName, lp.implementationVersion);

			if (strcmp(lp.layerName, "VK_LAYER_LUNARG_standard_validation") == 0)
			{
				enabledLayers.push_back("VK_LAYER_LUNARG_standard_validation");
			}
			if (strcmp(lp.layerName, "VK_LAYER_LUNARG_parameter_validation") == 0)
			{
				enabledLayers.push_back("VK_LAYER_LUNARG_parameter_validation");
			}
		}
#endif // ENABLE_VALIDATION_LAYER

		std::vector<const char*> enabledExtensions;

		uint32_t instanceExtensionPropertyCount;
		ret = vkEnumerateInstanceExtensionProperties(NULL, &instanceExtensionPropertyCount, NULL);
		if (ret != VK_SUCCESS)
		{
			fprintf(stderr, "vkEnumerateInstanceExtensionProperties failed %d\n", ret);
			return -1;
		}

		std::vector<VkExtensionProperties> instanceExtensionProperties(instanceExtensionPropertyCount);
		ret = vkEnumerateInstanceExtensionProperties(NULL, &instanceExtensionPropertyCount, instanceExtensionProperties.data());
		if (ret != VK_SUCCESS)
		{
			fprintf(stderr, "vkEnumerateInstanceExtensionProperties failed %d\n", ret);
			return -1;
		}

		support_VK_KHR_get_physical_device_properties2 = 0;
		support_VK_EXT_debug_utils = 0;
		for (uint32_t j = 0; j < instanceExtensionPropertyCount; j++)
		{
			const VkExtensionProperties& exp = instanceExtensionProperties[j];

			if (strcmp(exp.extensionName, "VK_KHR_get_physical_device_properties2") == 0)
				support_VK_KHR_get_physical_device_properties2 = exp.specVersion;
			if (strcmp(exp.extensionName, "VK_EXT_debug_utils") == 0)
				support_VK_EXT_debug_utils = exp.specVersion;
		}

		if (support_VK_KHR_get_physical_device_properties2)
			enabledExtensions.push_back("VK_KHR_get_physical_device_properties2");
#if ENABLE_VALIDATION_LAYER
		if (support_VK_EXT_debug_utils)
			enabledExtensions.push_back("VK_EXT_debug_utils");
#endif // ENABLE_VALIDATION_LAYER

		VkApplicationInfo applicationInfo;
		applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		applicationInfo.pNext = 0;
		applicationInfo.pApplicationName = "iml";
		applicationInfo.applicationVersion = 0;
		applicationInfo.pEngineName = "iml";
		applicationInfo.engineVersion = 20190319;
		applicationInfo.apiVersion = VK_MAKE_VERSION(1, 0, 0);

		VkInstanceCreateInfo instanceCreateInfo;
		instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		instanceCreateInfo.pNext = 0;
		instanceCreateInfo.flags = 0;
		instanceCreateInfo.pApplicationInfo = &applicationInfo;
		instanceCreateInfo.enabledLayerCount = enabledLayers.size();
		instanceCreateInfo.ppEnabledLayerNames = enabledLayers.data();
		instanceCreateInfo.enabledExtensionCount = enabledExtensions.size();
		instanceCreateInfo.ppEnabledExtensionNames = enabledExtensions.data();

		ret = vkCreateInstance(&instanceCreateInfo, 0, &g_instance);
		if (ret != VK_SUCCESS)
		{
			fprintf(stderr, "vkCreateInstance failed %d\n", ret);
			return -1;
		}

		return 0;
	}

	static int init_instance_extension()
	{
		if (support_VK_KHR_get_physical_device_properties2)
		{
			vkGetPhysicalDeviceFeatures2KHR = (PFN_vkGetPhysicalDeviceFeatures2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceFeatures2KHR");
			vkGetPhysicalDeviceProperties2KHR = (PFN_vkGetPhysicalDeviceProperties2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceProperties2KHR");
			vkGetPhysicalDeviceFormatProperties2KHR = (PFN_vkGetPhysicalDeviceFormatProperties2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceFormatProperties2KHR");
			vkGetPhysicalDeviceImageFormatProperties2KHR = (PFN_vkGetPhysicalDeviceImageFormatProperties2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceImageFormatProperties2KHR");
			vkGetPhysicalDeviceQueueFamilyProperties2KHR = (PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceQueueFamilyProperties2KHR");
			vkGetPhysicalDeviceMemoryProperties2KHR = (PFN_vkGetPhysicalDeviceMemoryProperties2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceMemoryProperties2KHR");
			vkGetPhysicalDeviceSparseImageFormatProperties2KHR = (PFN_vkGetPhysicalDeviceSparseImageFormatProperties2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceSparseImageFormatProperties2KHR");
		}

		return 0;
	}

	static int enum_physical_device(std::vector<VkPhysicalDevice>& physicalDevices) {
		physicalDevices.clear();
		uint32_t physicalDeviceCount = 0;
		int ret = vkEnumeratePhysicalDevices(g_instance, &physicalDeviceCount, 0);
		if (ret != VK_SUCCESS)
		{
			fprintf(stderr, "vkEnumeratePhysicalDevices failed %d\n", ret);
			return -1;
		}

		if (physicalDeviceCount > NCNN_MAX_GPU_COUNT)
			physicalDeviceCount = NCNN_MAX_GPU_COUNT;

		physicalDevices.resize(physicalDeviceCount);

		ret = vkEnumeratePhysicalDevices(g_instance, &physicalDeviceCount, physicalDevices.data());
		if (ret != VK_SUCCESS)
		{
			fprintf(stderr, "vkEnumeratePhysicalDevices failed %d\n", ret);
			return -1;
		}

		return 0;
	}

	static uint32_t find_device_compute_queue(const std::vector<VkQueueFamilyProperties>& queueFamilyProperties)
	{
		// first try, compute only queue
		for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
		{
			const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

			if ((queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT)
				&& !(queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT))
			{
				return i;
			}
		}

		// second try, any queue with compute
		for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
		{
			const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

			if (queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT)
			{
				return i;
			}
		}

		//     fprintf(stderr, "no compute queue\n");
		return -1;
	}

	static uint32_t find_device_transfer_queue(const std::vector<VkQueueFamilyProperties>& queueFamilyProperties)
	{
		// first try, transfer only queue
		for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
		{
			const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

			if ((queueFamilyProperty.queueFlags & VK_QUEUE_TRANSFER_BIT)
				&& !(queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT)
				&& !(queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT))
			{
				return i;
			}
		}

		// second try, any queue with transfer
		for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
		{
			const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

			if (queueFamilyProperty.queueFlags & VK_QUEUE_TRANSFER_BIT)
			{
				return i;
			}
		}

		// third try, use compute queue
		uint32_t compute_queue_index = find_device_compute_queue(queueFamilyProperties);
		if (compute_queue_index != (uint32_t)-1)
		{
			return compute_queue_index;
		}

		//     fprintf(stderr, "no transfer queue\n");
		return -1;
	}

	static uint32_t find_unified_memory(VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties)
	{
		// first try, host visible + host coherent + device local
		for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++)
		{
			const VkMemoryType& memoryType = physicalDeviceMemoryProperties.memoryTypes[i];

			if ((memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
				&& (memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
				&& (memoryType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
			{
				return i;
			}
		}

		// second try, host visible + device local
		for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++)
		{
			const VkMemoryType& memoryType = physicalDeviceMemoryProperties.memoryTypes[i];

			if ((memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
				&& (memoryType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
			{
				return i;
			}
		}

		//     fprintf(stderr, "no unified memory\n");
		return -1;
	}

	static uint32_t find_device_local_memory(VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties)
	{
		// first try, device local only
		for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++)
		{
			const VkMemoryType& memoryType = physicalDeviceMemoryProperties.memoryTypes[i];

			if (memoryType.propertyFlags == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
			{
				return i;
			}
		}

		// second try, with device local bit
		for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++)
		{
			const VkMemoryType& memoryType = physicalDeviceMemoryProperties.memoryTypes[i];

			if (memoryType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
			{
				return i;
			}
		}

		//     fprintf(stderr, "no device local memory\n");
		return -1;
	}

	static uint32_t find_host_visible_memory(VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties)
	{
		// first try, host visible + host coherent, without device local bit
		for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++)
		{
			const VkMemoryType& memoryType = physicalDeviceMemoryProperties.memoryTypes[i];

			if ((memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
				&& (memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
				&& !(memoryType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
			{
				return i;
			}
		}

		// second try, with host visible bit, without device local bit
		for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++)
		{
			const VkMemoryType& memoryType = physicalDeviceMemoryProperties.memoryTypes[i];

			if ((memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
				&& !(memoryType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
			{
				return i;
			}
		}

		// third try, with host visible bit
		for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++)
		{
			const VkMemoryType& memoryType = physicalDeviceMemoryProperties.memoryTypes[i];

			if (memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
			{
				return i;
			}
		}

		//     fprintf(stderr, "no host visible memory\n");
		return -1;
	}

	static int get_gpu_info(std::vector<VkPhysicalDevice>& physicalDevices) {
		uint32_t physicalDeviceCount = physicalDevices.size();
		// find proper device and queue
		int gpu_info_index = 0;
		int ret = 0;
		for (uint32_t i = 0; i < physicalDeviceCount; i++)
		{
			const VkPhysicalDevice& physicalDevice = physicalDevices[i];
			GpuInfo& gpu_info = g_gpu_infos[gpu_info_index];

			// device type
			VkPhysicalDeviceProperties physicalDeviceProperties;
			vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

			if (physicalDeviceProperties.vendorID == 0x13b5 && physicalDeviceProperties.apiVersion < VK_MAKE_VERSION(1, 0, 66))
			{
				// ignore arm mali with old buggy driver
				fprintf(stderr, "arm mali driver is too old\n");
				continue;
			}

			if (physicalDeviceProperties.vendorID == 0x5143 && physicalDeviceProperties.apiVersion < VK_MAKE_VERSION(1, 0, 49))
			{
				// ignore qcom adreno with old buggy driver
				fprintf(stderr, "qcom adreno driver is too old\n");
				continue;
			}

			gpu_info.physical_device = physicalDevice;

			// info
			gpu_info.api_version = physicalDeviceProperties.apiVersion;
			gpu_info.driver_version = physicalDeviceProperties.driverVersion;
			gpu_info.vendor_id = physicalDeviceProperties.vendorID;
			gpu_info.device_id = physicalDeviceProperties.deviceID;
			memcpy(gpu_info.pipeline_cache_uuid, physicalDeviceProperties.pipelineCacheUUID, VK_UUID_SIZE);

			if (physicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
				gpu_info.type = 0;
			else if (physicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
				gpu_info.type = 1;
			else if (physicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU)
				gpu_info.type = 2;
			else if (physicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)
				gpu_info.type = 3;
			else
				gpu_info.type = -1;

			// device capability
			gpu_info.max_shared_memory_size = physicalDeviceProperties.limits.maxComputeSharedMemorySize;

			gpu_info.max_workgroup_count[0] = physicalDeviceProperties.limits.maxComputeWorkGroupCount[0];
			gpu_info.max_workgroup_count[1] = physicalDeviceProperties.limits.maxComputeWorkGroupCount[1];
			gpu_info.max_workgroup_count[2] = physicalDeviceProperties.limits.maxComputeWorkGroupCount[2];

			gpu_info.max_workgroup_invocations = physicalDeviceProperties.limits.maxComputeWorkGroupInvocations;

			gpu_info.max_workgroup_size[0] = physicalDeviceProperties.limits.maxComputeWorkGroupSize[0];
			gpu_info.max_workgroup_size[1] = physicalDeviceProperties.limits.maxComputeWorkGroupSize[1];
			gpu_info.max_workgroup_size[2] = physicalDeviceProperties.limits.maxComputeWorkGroupSize[2];

			gpu_info.memory_map_alignment = physicalDeviceProperties.limits.minMemoryMapAlignment;
			gpu_info.buffer_offset_alignment = physicalDeviceProperties.limits.minStorageBufferOffsetAlignment;

			gpu_info.timestamp_period = physicalDeviceProperties.limits.timestampPeriod;

			//         fprintf(stderr, "[%u] max_shared_memory_size = %u\n", i, gpu_info.max_shared_memory_size);
			//         fprintf(stderr, "[%u] max_workgroup_count = %u %u %u\n", i, gpu_info.max_workgroup_count[0], gpu_info.max_workgroup_count[1], gpu_info.max_workgroup_count[2]);
			//         fprintf(stderr, "[%u] max_workgroup_invocations = %u\n", i, gpu_info.max_workgroup_invocations);
			//         fprintf(stderr, "[%u] max_workgroup_size = %u %u %u\n", i, gpu_info.max_workgroup_size[0], gpu_info.max_workgroup_size[1], gpu_info.max_workgroup_size[2]);
			//         fprintf(stderr, "[%u] memory_map_alignment = %lu\n", i, gpu_info.memory_map_alignment);
			//         fprintf(stderr, "[%u] buffer_offset_alignment = %lu\n", i, gpu_info.buffer_offset_alignment);

					// find compute queue
			uint32_t queueFamilyPropertiesCount;
			vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertiesCount, 0);

			std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyPropertiesCount);
			vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertiesCount, queueFamilyProperties.data());

			gpu_info.compute_queue_family_index = find_device_compute_queue(queueFamilyProperties);
			gpu_info.transfer_queue_family_index = find_device_transfer_queue(queueFamilyProperties);

			gpu_info.compute_queue_count = queueFamilyProperties[gpu_info.compute_queue_family_index].queueCount;
			gpu_info.transfer_queue_count = queueFamilyProperties[gpu_info.transfer_queue_family_index].queueCount;

			// find memory type index
			VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;
			vkGetPhysicalDeviceMemoryProperties(physicalDevice, &physicalDeviceMemoryProperties);

			gpu_info.unified_memory_index = find_unified_memory(physicalDeviceMemoryProperties);
			gpu_info.device_local_memory_index = find_device_local_memory(physicalDeviceMemoryProperties);
			gpu_info.host_visible_memory_index = find_host_visible_memory(physicalDeviceMemoryProperties);

			// treat as unified memory architecture if memory heap is the same
			if (gpu_info.unified_memory_index != (uint32_t)-1)
			{
				int unified_memory_heap_index = physicalDeviceMemoryProperties.memoryTypes[gpu_info.unified_memory_index].heapIndex;
				int device_local_memory_heap_index = physicalDeviceMemoryProperties.memoryTypes[gpu_info.device_local_memory_index].heapIndex;
				int host_visible_memory_heap_index = physicalDeviceMemoryProperties.memoryTypes[gpu_info.host_visible_memory_index].heapIndex;
				if (unified_memory_heap_index == device_local_memory_heap_index && unified_memory_heap_index == host_visible_memory_heap_index)
				{
					gpu_info.device_local_memory_index = gpu_info.unified_memory_index;
					gpu_info.host_visible_memory_index = gpu_info.unified_memory_index;
				}
			}

			// get device extension
			uint32_t deviceExtensionPropertyCount = 0;
			ret = vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &deviceExtensionPropertyCount, NULL);
			if (ret != VK_SUCCESS)
			{
				fprintf(stderr, "vkEnumerateDeviceExtensionProperties failed %d\n", ret);
				return -1;
			}

			std::vector<VkExtensionProperties> deviceExtensionProperties(deviceExtensionPropertyCount);
			ret = vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &deviceExtensionPropertyCount, deviceExtensionProperties.data());
			if (ret != VK_SUCCESS)
			{
				fprintf(stderr, "vkEnumerateDeviceExtensionProperties failed %d\n", ret);
				return -1;
			}

			// extension capability
			gpu_info.support_VK_KHR_8bit_storage = 0;
			gpu_info.support_VK_KHR_16bit_storage = 0;
			gpu_info.support_VK_KHR_bind_memory2 = 0;
			gpu_info.support_VK_KHR_dedicated_allocation = 0;
			gpu_info.support_VK_KHR_descriptor_update_template = 0;
			gpu_info.support_VK_KHR_get_memory_requirements2 = 0;
			gpu_info.support_VK_KHR_push_descriptor = 0;
			gpu_info.support_VK_KHR_shader_float16_int8 = 0;
			gpu_info.support_VK_KHR_shader_float_controls = 0;
			gpu_info.support_VK_KHR_storage_buffer_storage_class = 0;
			for (uint32_t j = 0; j < deviceExtensionPropertyCount; j++)
			{
				const VkExtensionProperties& exp = deviceExtensionProperties[j];
				//             fprintf(stderr, "device extension %s = %u\n", exp.extensionName, exp.specVersion);

				if (strcmp(exp.extensionName, "VK_KHR_8bit_storage") == 0)
					gpu_info.support_VK_KHR_8bit_storage = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_16bit_storage") == 0)
					gpu_info.support_VK_KHR_16bit_storage = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_bind_memory2") == 0)
					gpu_info.support_VK_KHR_bind_memory2 = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_dedicated_allocation") == 0)
					gpu_info.support_VK_KHR_dedicated_allocation = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_descriptor_update_template") == 0)
					gpu_info.support_VK_KHR_descriptor_update_template = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_get_memory_requirements2") == 0)
					gpu_info.support_VK_KHR_get_memory_requirements2 = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_push_descriptor") == 0)
					gpu_info.support_VK_KHR_push_descriptor = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_shader_float16_int8") == 0)
					gpu_info.support_VK_KHR_shader_float16_int8 = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_shader_float_controls") == 0)
					gpu_info.support_VK_KHR_shader_float_controls = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_storage_buffer_storage_class") == 0)
					gpu_info.support_VK_KHR_storage_buffer_storage_class = exp.specVersion;
			}

			// check features
			gpu_info.support_fp16_packed = true;
			gpu_info.support_fp16_storage = false;
			gpu_info.support_fp16_arithmetic = false;
			gpu_info.support_int8_storage = false;
			gpu_info.support_int8_arithmetic = false;
			if (support_VK_KHR_get_physical_device_properties2)
			{
				void* queryExtensionFeatures = 0;

				// query int8 storage
				VkPhysicalDevice8BitStorageFeaturesKHR query8BitStorageFeatures;
				query8BitStorageFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR;
				query8BitStorageFeatures.pNext = 0;
				if (gpu_info.support_VK_KHR_8bit_storage)
				{
					query8BitStorageFeatures.pNext = queryExtensionFeatures;
					queryExtensionFeatures = &query8BitStorageFeatures;
				}

				// query fp16/int16 storage
				VkPhysicalDevice16BitStorageFeaturesKHR query16BitStorageFeatures;
				query16BitStorageFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR;
				query16BitStorageFeatures.pNext = 0;
				if (gpu_info.support_VK_KHR_16bit_storage)
				{
					query16BitStorageFeatures.pNext = queryExtensionFeatures;
					queryExtensionFeatures = &query16BitStorageFeatures;
				}

				// query fp16/int8 arithmetic
				VkPhysicalDeviceFloat16Int8FeaturesKHR queryFloat16Int8Features;
				queryFloat16Int8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR;
				queryFloat16Int8Features.pNext = 0;
				if (gpu_info.support_VK_KHR_shader_float16_int8)
				{
					queryFloat16Int8Features.pNext = queryExtensionFeatures;
					queryExtensionFeatures = &queryFloat16Int8Features;
				}

				VkPhysicalDeviceFeatures2KHR queryFeatures;
				queryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR,
					queryFeatures.pNext = queryExtensionFeatures;

				vkGetPhysicalDeviceFeatures2KHR(physicalDevice, &queryFeatures);

				if (gpu_info.support_VK_KHR_8bit_storage)
				{
					gpu_info.support_int8_storage = query8BitStorageFeatures.storageBuffer8BitAccess && query8BitStorageFeatures.uniformAndStorageBuffer8BitAccess;
				}
				if (gpu_info.support_VK_KHR_16bit_storage)
				{
					gpu_info.support_fp16_storage = query16BitStorageFeatures.storageBuffer16BitAccess && query16BitStorageFeatures.uniformAndStorageBuffer16BitAccess;
				}
				if (gpu_info.support_VK_KHR_shader_float16_int8)
				{
					gpu_info.support_fp16_arithmetic = queryFloat16Int8Features.shaderFloat16;
					gpu_info.support_int8_arithmetic = queryFloat16Int8Features.shaderInt8;
				}
			}
			else
			{
				//             // TODO
				//             VkPhysicalDeviceFeatures features;
				//             vkGetPhysicalDeviceFeatures(physicalDevice, &features);
			}

			if (physicalDeviceProperties.vendorID == 0x13b5)
			{
				// the 16bit_storage implementation of arm mali driver is buggy :[
				gpu_info.support_fp16_storage = false;
			}

			fprintf(stderr, "[%u %s]  queueC=%u[%u]  queueT=%u[%u]  memU=%u  memDL=%u  memHV=%u\n", i, physicalDeviceProperties.deviceName,
				gpu_info.compute_queue_family_index, gpu_info.compute_queue_count,
				gpu_info.transfer_queue_family_index, gpu_info.transfer_queue_count,
				gpu_info.unified_memory_index, gpu_info.device_local_memory_index, gpu_info.host_visible_memory_index);

			fprintf(stderr, "[%u %s]  fp16p=%d  fp16s=%d  fp16a=%d  int8s=%d  int8a=%d\n", i, physicalDeviceProperties.deviceName,
				gpu_info.support_fp16_packed, gpu_info.support_fp16_storage, gpu_info.support_fp16_arithmetic,
				gpu_info.support_int8_storage, gpu_info.support_int8_arithmetic);

			gpu_info_index++;
		}

		g_gpu_count = gpu_info_index;

		return ret;
	}

	static int find_default_vulkan_device_index()
	{
		// first try, ¶ÀÁ¢ÏÔ¿¨
		for (int i = 0; i < g_gpu_count; i++)
		{
			if (g_gpu_infos[i].type == 0)
				return i;
		}

		// second try, ¼¯³ÉÏÔ¿¨
		for (int i = 0; i < g_gpu_count; i++)
		{
			if (g_gpu_infos[i].type == 1)
				return i;
		}

		// third try, any probed device
		if (g_gpu_count > 0)
			return 0;

		fprintf(stderr, "no vulkan device\n");
		return -1;
	}

	int create_gpu_instance() {
		int ret = create_instance();
		if (ret) {
			return ret;
		}
		init_instance_extension();

		std::vector<VkPhysicalDevice> physicalDevices;
		ret = enum_physical_device(physicalDevices);
		if (ret) {
			return ret;
		}

		ret = get_gpu_info(physicalDevices);
		if (ret) {
			return ret;
		}

		// the default gpu device
		g_default_gpu_index = find_default_vulkan_device_index();

		return 0;
	}

	void destroy_gpu_instance() {
		for (int i = 0; i < NCNN_MAX_GPU_COUNT; i++)
		{
			delete g_default_vkdev[i];
			g_default_vkdev[i] = 0;
		}

		vkDestroyInstance(g_instance, 0);
	}

	int get_default_gpu_index() {
		return g_default_gpu_index;
	}

	int get_gpu_count() {
		return g_gpu_count;
	}
}
}