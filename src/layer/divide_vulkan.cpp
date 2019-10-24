#include "divide_vulkan.h"

namespace iml {
namespace train {

	Divide_vulkan::Divide_vulkan(float div_num) : _div_num(div_num)
	{
		pipeline_relu = 0;
	}

	Divide_vulkan::~Divide_vulkan()
	{
		pipeline_relu = 0;
	}

	int Divide_vulkan::create_pipeline(const VulkanDevice* vkdev)
	{
		std::vector<vk_specialization_type> specializations(1);
		specializations[0].f = _div_num;
		// pack1
		{
			//glsl÷–binding
			std::vector<VkDescriptorType> bufferTypes = {
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
			};
			pipeline_relu = new Pipeline(vkdev);
			pipeline_relu->set_optimal_local_size_xyz();
			pipeline_relu->create("divide", specializations, bufferTypes, 5);
		}

		return 0;
	}

	int Divide_vulkan::destroy_pipeline()
	{
		delete pipeline_relu;
		pipeline_relu = 0;

		return 0;
	}

	int Divide_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd) const
	{
		// binding in glsl
		std::vector<VkMat> bindings(1);
		bindings[0] = bottom_top_blob;

		// push_constant in glsl
		std::vector<vk_constant_type> constants(5);
		constants[0].i = bottom_top_blob.dims;
		constants[1].i = bottom_top_blob.w;
		constants[2].i = bottom_top_blob.h;
		constants[3].i = bottom_top_blob.c;
		constants[4].i = bottom_top_blob.cstep;

		const Pipeline* pipeline = pipeline_relu;

		cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);

		return 0;
	}

}
} // namespace ncnn
