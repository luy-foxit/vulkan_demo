#include "resize_vulkan.h"

namespace iml {
namespace train {

	Resize_vulkan::Resize_vulkan()
	{
		pipeline_resize = 0;
	}

	Resize_vulkan::~Resize_vulkan()
	{
		pipeline_resize = 0;
	}

	int Resize_vulkan::create_pipeline(const VulkanDevice* vkdev)
	{
		std::vector<vk_specialization_type> specializations(1);
		specializations[0].i = 0;
		// pack1
		{
			int binding_count = 2;	//glsl中binding数量
			int push_constant_count = 10;	//glsl中push_constant参数数量

			pipeline_resize = new Pipeline(vkdev);
			pipeline_resize->set_optimal_local_size_xyz();
			pipeline_resize->create("resize", specializations, binding_count, push_constant_count);
		}

		return 0;
	}

	int Resize_vulkan::destroy_pipeline()
	{
		delete pipeline_resize;
		pipeline_resize = 0;

		return 0;
	}

	int Resize_vulkan::forward(VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd) const
	{
		// binding in glsl
		std::vector<VkMat> bindings(2);
		bindings[0] = bottom_blob;
		bindings[1] = top_blob;

		// push_constant in glsl
		std::vector<vk_constant_type> constants(10);
		constants[0].i = bottom_blob.dims;
		constants[1].i = bottom_blob.w;
		constants[2].i = bottom_blob.h;
		constants[3].i = bottom_blob.c;
		constants[4].i = bottom_blob.cstep;

		constants[5].i = top_blob.dims;
		constants[6].i = top_blob.w;
		constants[7].i = top_blob.h;
		constants[8].i = top_blob.c;
		constants[9].i = top_blob.cstep;

		const Pipeline* pipeline = pipeline_resize;

		cmd.record_pipeline(pipeline, bindings, constants, top_blob);

		return 0;
	}

}
} // namespace ncnn
