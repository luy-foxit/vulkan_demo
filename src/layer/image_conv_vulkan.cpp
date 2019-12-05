#include "image_conv_vulkan.h"


namespace iml {
namespace train {

	ImageConv_vulkan::ImageConv_vulkan()
	{
		pipeline_convolution = 0;
	}

	ImageConv_vulkan::~ImageConv_vulkan()
	{
		pipeline_convolution = 0;
	}

	int ImageConv_vulkan::create_pipeline(const VulkanDevice* vkdev)
	{
		//constant_id 数量
		std::vector<vk_specialization_type> specializations(7);
		specializations[0].i = 3;		//kernel_w in glsl
		specializations[1].i = 3;		//kernel_h in glsl
		specializations[2].i = 1;		//dilation_w in glsl
		specializations[3].i = 1;		//dilation_h in glsl
		specializations[4].i = 1;		//stride_w in glsl
		specializations[5].i = 1;		//stride_h in glsl
		specializations[6].i = 1;		//bias_term in glsl

		{
			//glsl中binding
			std::vector<VkDescriptorType> bufferTypes = {
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
			};

			int push_constant_count = 10;	//glsl中push_constant参数数量

			pipeline_convolution = new Pipeline(vkdev);
			pipeline_convolution->set_optimal_local_size_xyz();
			pipeline_convolution->create("image_conv", specializations, bufferTypes, push_constant_count);
		}

		return 0;
	}

	int ImageConv_vulkan::destroy_pipeline()
	{
		delete pipeline_convolution;
		pipeline_convolution = 0;

		return 0;
	}

	int ImageConv_vulkan::upload_model(VkTransfer& cmd, std::vector<float>& weight_data, std::vector<float>& bias_data) {
		cmd.record_upload(weight_data, weight_data_gpu);
		cmd.record_upload(bias_data, bias_data_gpu);

		return 0;
	}

	int ImageConv_vulkan::forward(VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, Option& opt) const
	{
		int elemsize = sizeof(float);
		int elempack = 1;
		top_blob.create(bottom_blob.w, bottom_blob.h, 8, elemsize, elempack, opt.blob_vkallocator, opt.staging_vkallocator);

		// binding in glsl
		std::vector<VkMat> bindings(4);
		bindings[0] = bottom_blob;
		bindings[0] = top_blob;
		bindings[1] = weight_data_gpu;	//weights
		bindings[2] = bias_data_gpu;	//bias

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

		const Pipeline* pipeline = pipeline_convolution;

		cmd.record_pipeline(pipeline, bindings, constants, top_blob);

		return 0;
	}

}
} // namespace ncnn
