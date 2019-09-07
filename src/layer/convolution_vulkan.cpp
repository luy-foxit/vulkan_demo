#include "convolution_vulkan.h"

namespace iml {
namespace train {

	Convolution_vulkan::Convolution_vulkan()
	{
		pipeline_convolution = 0;
	}

	Convolution_vulkan::~Convolution_vulkan()
	{
		pipeline_convolution = 0;
	}

	int Convolution_vulkan::create_pipeline(const VulkanDevice* vkdev)
	{
		//constant_id 数量
		std::vector<vk_specialization_type> specializations(3);
		specializations[0].i = 3;		//kernel_w in glsl
		specializations[1].i = 3;		//kernel_h in glsl
		specializations[2].i = 1;		//bias_term in glsl

		{
			int binding_count = 3;	//glsl中binding数量
			int push_constant_count = 5;	//glsl中push_constant参数数量

			pipeline_convolution = new Pipeline(vkdev);
			pipeline_convolution->set_optimal_local_size_xyz();
			pipeline_convolution->create("convolution", specializations, binding_count, push_constant_count);
		}

		return 0;
	}

	int Convolution_vulkan::destroy_pipeline()
	{
		delete pipeline_convolution;
		pipeline_convolution = 0;

		return 0;
	}

	int Convolution_vulkan::upload_model(VkTransfer& cmd, std::vector<float>& weight_data, std::vector<float>& bias_data) {
		cmd.record_upload(weight_data, weight_data_gpu);
		cmd.record_upload(bias_data, bias_data_gpu);

		return 0;
	}

	int Convolution_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd) const
	{
		// binding in glsl
		std::vector<VkMat> bindings(3);
		bindings[0] = bottom_top_blob;
		bindings[1] = weight_data_gpu;	//weights
		bindings[2] = bias_data_gpu;	//bias

		// push_constant in glsl
		std::vector<vk_constant_type> constants(5);
		constants[0].i = bottom_top_blob.dims;
		constants[1].i = bottom_top_blob.w;
		constants[2].i = bottom_top_blob.h;
		constants[3].i = bottom_top_blob.c;
		constants[4].i = bottom_top_blob.cstep;

		const Pipeline* pipeline = pipeline_convolution;

		cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);

		return 0;
	}

}
} // namespace ncnn
