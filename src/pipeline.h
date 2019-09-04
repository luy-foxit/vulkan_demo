#pragma once

#include <opencv2/opencv.hpp>
#include <vulkan/vulkan.h>
#include "gpu.h"

namespace iml {
namespace train {

	// type for vulkan specialization constant and push constant
	union vk_specialization_type { int i; float f; uint32_t u32; };
	union vk_constant_type { int i; float f; };

	class Option;
	class Pipeline
	{
	public:
		Pipeline(const VulkanDevice* vkdev);
		~Pipeline();

	public:
		void set_optimal_local_size_xyz(int w = 32, int h = 32, int c = 32);
		void set_local_size_xyz(int w, int h, int c);

		int create(const uint32_t* spv_data, size_t spv_data_size, const char* entry_name,
			const std::vector<vk_specialization_type>& specializations, int binding_count, int push_constant_count);
		int create(VkShaderModule shader_module, const char* entry_name,
			const std::vector<vk_specialization_type>& specializations, int binding_count, int push_constant_count);
		int create(const char* name, const Option& opt, const std::vector<vk_specialization_type>& specializations,
			int binding_count, int push_constant_count);
		void destroy();

	protected:
		int create_descriptorset_layout(int binding_count);
		int create_pipeline_layout(int push_constant_count);
		int create_pipeline(VkShaderModule shader_module, const char* entry_name, const std::vector<vk_specialization_type>& specializations);
		int create_descriptor_update_template(int binding_count);

	public:
		const VulkanDevice* vkdev;

		// local shader module
		VkShaderModule local_shader_module;

		VkDescriptorSetLayout descriptorset_layout;
		VkPipelineLayout pipeline_layout;

		// op forward TODO use pipeline cache ?
		VkPipeline pipeline;

		VkDescriptorUpdateTemplateKHR descriptor_update_template;

		uint32_t local_size_x;
		uint32_t local_size_y;
		uint32_t local_size_z;
	};

}
} // namespace ncnn

