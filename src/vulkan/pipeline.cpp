#include "pipeline.h"
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <string>

namespace iml {
namespace train {

	Pipeline::Pipeline(const VulkanDevice* _vkdev) : vkdev(_vkdev)
	{
		local_shader_module = 0;

		descriptorset_layout = 0;
		pipeline_layout = 0;
		pipeline = 0;
		descriptor_update_template = 0;

		local_size_x = 1;
		local_size_y = 1;
		local_size_z = 1;
	}

	Pipeline::~Pipeline()
	{
		destroy();
	}

	int Pipeline::create(const uint32_t* spv_data, size_t spv_data_size, const char* entry_name, const std::vector<vk_specialization_type>& specializations, const std::vector<VkDescriptorType>& bufferTypes, int push_constant_count)
	{
		local_shader_module = vkdev->compile_shader_module(spv_data, spv_data_size);

		//     fprintf(stderr, "local_shader_module %p %s created\n", local_shader_module, entry_name);

		return create(local_shader_module, entry_name, specializations, bufferTypes, push_constant_count);
	}

	int Pipeline::create(VkShaderModule shader_module, const char* entry_name, const std::vector<vk_specialization_type>& specializations, const std::vector<VkDescriptorType>& bufferTypes, int push_constant_count)
	{
		// 创建descript layout 到 descriptorset_layout, 作为 pipeline_layout的参数
		create_descriptorset_layout(bufferTypes);

		// 创建 pipeline layout 到 pipeline_layout, 作为 pipeline 的参数
		create_pipeline_layout(push_constant_count);

		// 创建pipeline
		create_pipeline(shader_module, entry_name, specializations);

		if (vkdev->info.support_VK_KHR_descriptor_update_template)
		{
			// 创建描述符更新模板到 descriptor_update_template
			create_descriptor_update_template(bufferTypes);
		}

		return 0;
	}

	int Pipeline::create(const char* _name, const std::vector<vk_specialization_type>& specializations, const std::vector<VkDescriptorType>& bufferTypes, int push_constant_count)
	{
		std::string name = _name;

		VkShaderModule shader_module = vkdev->get_shader_module(name.c_str());

		return create(shader_module, name.c_str(), specializations, bufferTypes, push_constant_count);
	}

	void Pipeline::destroy()
	{
		if (vkdev->info.support_VK_KHR_descriptor_update_template)
		{
			if (descriptor_update_template)
			{
				vkdev->vkDestroyDescriptorUpdateTemplateKHR(vkdev->vkdevice(), descriptor_update_template, 0);
				descriptor_update_template = 0;
			}
		}

		if (pipeline)
		{
			vkDestroyPipeline(vkdev->vkdevice(), pipeline, 0);
			pipeline = 0;
		}

		if (pipeline_layout)
		{
			vkDestroyPipelineLayout(vkdev->vkdevice(), pipeline_layout, 0);
			pipeline_layout = 0;
		}

		if (descriptorset_layout)
		{
			vkDestroyDescriptorSetLayout(vkdev->vkdevice(), descriptorset_layout, 0);
			descriptorset_layout = 0;
		}

		if (local_shader_module)
		{
			vkDestroyShaderModule(vkdev->vkdevice(), local_shader_module, 0);
			local_shader_module = 0;
		}
	}

	void Pipeline::set_optimal_local_size_xyz(int w, int h, int c)
	{
		if (c > 0)
		{
			local_size_z = vkdev->info.max_workgroup_size[2];
			while ((uint32_t)c < local_size_z)
			{
				local_size_z /= 2;
			}
		}
		else
		{
			local_size_z = std::min((uint32_t)128, vkdev->info.max_workgroup_size[2]);
		}

		uint32_t max_local_size_xy = vkdev->info.max_workgroup_invocations / local_size_z;

		if (h == w || (h < 0 && w < 0))
		{
			uint32_t local_size_xy = sqrt(max_local_size_xy);
			uint32_t local_size_xy_prefer = 128;
			while (local_size_xy < local_size_xy_prefer)
			{
				local_size_xy_prefer /= 2;
			}
			local_size_x = local_size_xy_prefer;
			local_size_y = local_size_xy_prefer;
		}
		if (h > 0 && w > 0)
		{
			if (h > w)
			{
				float ps = h / (float)w;
				float local_size_xy = sqrt(max_local_size_xy / ps);
				local_size_y = local_size_xy * ps;
				local_size_x = std::max((uint32_t)local_size_xy, (uint32_t)1);
			}
			else
			{
				float ps = w / (float)h;
				float local_size_xy = sqrt(max_local_size_xy / ps);
				local_size_y = std::max((uint32_t)local_size_xy, (uint32_t)1);
				local_size_x = local_size_xy * ps;
			}

			uint32_t local_size_y_prefer = std::min((uint32_t)128, vkdev->info.max_workgroup_size[1]);
			while (local_size_y < local_size_y_prefer)
			{
				local_size_y_prefer /= 2;
			}

			uint32_t local_size_x_prefer = std::min((uint32_t)128, vkdev->info.max_workgroup_size[0]);
			while (local_size_x < local_size_x_prefer)
			{
				local_size_x_prefer /= 2;
			}

			local_size_y = local_size_y_prefer;
			local_size_x = local_size_x_prefer;
		}
		else if (h > 0)
		{
			local_size_y = std::min(max_local_size_xy, vkdev->info.max_workgroup_size[1]);
			while ((uint32_t)h < local_size_y)
			{
				local_size_y /= 2;
			}

			uint32_t max_local_size_x = max_local_size_xy / local_size_y;
			local_size_x = std::min(max_local_size_x, vkdev->info.max_workgroup_size[0]);
		}
		else if (w > 0)
		{
			local_size_x = std::min(max_local_size_xy, vkdev->info.max_workgroup_size[0]);
			while ((uint32_t)w < local_size_x)
			{
				local_size_x /= 2;
			}

			uint32_t max_local_size_y = max_local_size_xy / local_size_x;
			local_size_y = std::min(max_local_size_y, vkdev->info.max_workgroup_size[1]);
		}

		//     fprintf(stderr, "local size = %d %d %d\n", local_size_x, local_size_y, local_size_z);
	}

	void Pipeline::set_local_size_xyz(int w, int h, int c)
	{
		local_size_x = w;
		local_size_y = h;
		local_size_z = c;
	}

	// 创建描述符集布局
	int Pipeline::create_descriptorset_layout(const std::vector<VkDescriptorType>& bufferTypes)
	{
		int binding_count = bufferTypes.size();
		if (binding_count == 0)
		{
			descriptorset_layout = 0;
			return 0;
		}

		//配置glsl中的binding
		std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings(binding_count);
		for (int i = 0; i < binding_count; i++)
		{
			auto type = bufferTypes[i];
			descriptorSetLayoutBindings[i].binding = i;
			descriptorSetLayoutBindings[i].descriptorType = type; //VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorSetLayoutBindings[i].descriptorCount = 1;
			descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			descriptorSetLayoutBindings[i].pImmutableSamplers = 0;
		}

		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo;
		descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		descriptorSetLayoutCreateInfo.pNext = 0;
		descriptorSetLayoutCreateInfo.flags = 0;
		descriptorSetLayoutCreateInfo.bindingCount = binding_count;
		descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings.data();

		if (vkdev->info.support_VK_KHR_push_descriptor)
		{
			descriptorSetLayoutCreateInfo.flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
		}

		VkResult ret = vkCreateDescriptorSetLayout(vkdev->vkdevice(), &descriptorSetLayoutCreateInfo, 0, &descriptorset_layout);
		if (ret != VK_SUCCESS)
		{
			fprintf(stderr, "vkCreateDescriptorSetLayout failed %d\n", ret);
			return -1;
		}

		return 0;
	}

	int Pipeline::create_pipeline_layout(int push_constant_count)
	{
		VkPushConstantRange pushConstantRange;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(int) * push_constant_count;

		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo;
		pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutCreateInfo.pNext = 0;
		pipelineLayoutCreateInfo.flags = 0;

		if (descriptorset_layout)
		{
			pipelineLayoutCreateInfo.setLayoutCount = 1;
			pipelineLayoutCreateInfo.pSetLayouts = &descriptorset_layout;
		}
		else
		{
			pipelineLayoutCreateInfo.setLayoutCount = 0;
			pipelineLayoutCreateInfo.pSetLayouts = 0;
		}

		if (push_constant_count > 0)
		{
			pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
			pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
		}
		else
		{
			pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
			pipelineLayoutCreateInfo.pPushConstantRanges = 0;
		}

		VkResult ret = vkCreatePipelineLayout(vkdev->vkdevice(), &pipelineLayoutCreateInfo, 0, &pipeline_layout);
		if (ret != VK_SUCCESS)
		{
			fprintf(stderr, "vkCreatePipelineLayout failed %d\n", ret);
			return -1;
		}

		return 0;
	}

	int Pipeline::create_pipeline(VkShaderModule shader_module,
		const char* entry_name,
		const std::vector<vk_specialization_type>& specializations)
	{
		const int specialization_count = specializations.size();

		// +3 for local_size_xyz
		std::vector<VkSpecializationMapEntry> specializationMapEntries;
		specializationMapEntries.resize(specialization_count + 3);

		for (int i = 0; i < specialization_count; i++)
		{
			// constant_id in glsl
			specializationMapEntries[i].constantID = i;
			specializationMapEntries[i].offset = i * sizeof(vk_specialization_type);
			specializationMapEntries[i].size = sizeof(vk_specialization_type);
		}

		std::vector<vk_specialization_type> specialization_data = specializations;

		// append local_size_xyz specialization
		{
			VkSpecializationMapEntry* local_size_xyz_entries = specializationMapEntries.data() + specialization_count;

			// local_size_x_id in glsl
			local_size_xyz_entries[0].constantID = 233;
			local_size_xyz_entries[0].offset = (specialization_count + 0) * sizeof(vk_specialization_type);
			local_size_xyz_entries[0].size = sizeof(vk_specialization_type);

			// local_size_y_id in glsl
			local_size_xyz_entries[1].constantID = 234;
			local_size_xyz_entries[1].offset = (specialization_count + 1) * sizeof(vk_specialization_type);
			local_size_xyz_entries[1].size = sizeof(vk_specialization_type);

			// local_size_z_id in glsl
			local_size_xyz_entries[2].constantID = 235;
			local_size_xyz_entries[2].offset = (specialization_count + 2) * sizeof(vk_specialization_type);
			local_size_xyz_entries[2].size = sizeof(vk_specialization_type);

			specialization_data.resize(specialization_count + 3);
			specialization_data[specialization_count + 0].u32 = local_size_x;
			specialization_data[specialization_count + 1].u32 = local_size_y;
			specialization_data[specialization_count + 2].u32 = local_size_z;
		}

		VkSpecializationInfo specializationInfo;
		specializationInfo.mapEntryCount = specializationMapEntries.size();
		specializationInfo.pMapEntries = specializationMapEntries.data();
		specializationInfo.dataSize = specialization_data.size() * sizeof(vk_specialization_type);
		specializationInfo.pData = specialization_data.data();

		VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo;
		pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		pipelineShaderStageCreateInfo.pNext = 0;
		pipelineShaderStageCreateInfo.flags = 0;
		pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		pipelineShaderStageCreateInfo.module = shader_module;
		//pipelineShaderStageCreateInfo.pName = entry_name;
		pipelineShaderStageCreateInfo.pName = "main";
		pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;

		VkComputePipelineCreateInfo computePipelineCreateInfo;
		computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		computePipelineCreateInfo.pNext = 0;
		computePipelineCreateInfo.flags = 0;
		computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
		computePipelineCreateInfo.layout = pipeline_layout;
		computePipelineCreateInfo.basePipelineHandle = 0;
		computePipelineCreateInfo.basePipelineIndex = 0;

		VkDevice phy_dev = vkdev->vkdevice();
		VkResult ret = vkCreateComputePipelines(phy_dev, 0, 1, &computePipelineCreateInfo, 0, &pipeline);
		if (ret != VK_SUCCESS)
		{
			fprintf(stderr, "vkCreateComputePipelines failed %d\n", ret);
			return -1;
		}

		return 0;
	}

	static inline size_t descripterTypeToStride(VkDescriptorType type) {
		size_t stride = 0;
		switch (type) {
		case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
		case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
		case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
		{
			stride = sizeof(VkDescriptorImageInfo);
			break;
		}
		case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
		case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
		case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
		case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
		{
			stride = sizeof(VkDescriptorBufferInfo);
			break;
		}
		}
		return stride;
	}

	int Pipeline::create_descriptor_update_template(const std::vector<VkDescriptorType>& bufferTypes)
	{
		int binding_count = bufferTypes.size();
		if (binding_count == 0)
		{
			descriptor_update_template = 0;
			return 0;
		}

		std::vector<VkDescriptorUpdateTemplateEntryKHR> descriptorUpdateTemplateEntries(binding_count);
        size_t offset = 0;
		for (int i = 0; i < binding_count; i++)// TODO do not update weights
		{
			auto type = bufferTypes[i];
			size_t stride = descripterTypeToStride(type);
			descriptorUpdateTemplateEntries[i].dstBinding = i;
			descriptorUpdateTemplateEntries[i].dstArrayElement = 0;
			descriptorUpdateTemplateEntries[i].descriptorCount = 1;
			descriptorUpdateTemplateEntries[i].descriptorType = type; //VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorUpdateTemplateEntries[i].offset = offset; //i * sizeof(VkDescriptorBufferInfo);
			descriptorUpdateTemplateEntries[i].stride = stride; //sizeof(VkDescriptorBufferInfo);
            offset += stride;
		}

		VkDescriptorUpdateTemplateCreateInfoKHR descriptorUpdateTemplateCreateInfo;
		descriptorUpdateTemplateCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO_KHR;
		descriptorUpdateTemplateCreateInfo.pNext = 0;
		descriptorUpdateTemplateCreateInfo.flags = 0;
		descriptorUpdateTemplateCreateInfo.descriptorUpdateEntryCount = binding_count;// TODO do not update weights
		descriptorUpdateTemplateCreateInfo.pDescriptorUpdateEntries = descriptorUpdateTemplateEntries.data();
		if (vkdev->info.support_VK_KHR_push_descriptor)
		{
			descriptorUpdateTemplateCreateInfo.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR;
		}
		else
		{
			descriptorUpdateTemplateCreateInfo.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET_KHR;
		}
		// descriptorSetLayout should be ignored if VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR
		// FIXME HACK WARNING TODO NOTE but crash on radv if set NULL  :(
		descriptorUpdateTemplateCreateInfo.descriptorSetLayout = descriptorset_layout;
		descriptorUpdateTemplateCreateInfo.pipelineBindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
		descriptorUpdateTemplateCreateInfo.pipelineLayout = pipeline_layout;
		descriptorUpdateTemplateCreateInfo.set = 0;

		VkResult ret = vkdev->vkCreateDescriptorUpdateTemplateKHR(vkdev->vkdevice(), &descriptorUpdateTemplateCreateInfo, 0, &descriptor_update_template);
		if (ret != VK_SUCCESS)
		{
			fprintf(stderr, "vkCreateDescriptorUpdateTemplateKHR failed %d\n", ret);
			return -1;
		}

		return 0;
	}

}
} // namespace ncnn
