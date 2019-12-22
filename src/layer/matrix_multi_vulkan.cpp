#include "matrix_multi_vulkan.h"


namespace iml {
namespace train {

	static void AddDot(int k, float *x, float *y, int n, float *gamma);

	MatrixMulti_vulkan::MatrixMulti_vulkan()
	{
		pipeline_mm = nullptr;
	}

	MatrixMulti_vulkan::~MatrixMulti_vulkan()
	{
		if (pipeline_mm) {
			delete pipeline_mm;
		}
	}

	int MatrixMulti_vulkan::create_pipeline(const VulkanDevice* vkdev)
	{
		std::vector<vk_specialization_type> specializations(1);
		specializations[0].i = 0;
		// pack1
		{
			//glsl中binding
			std::vector<VkDescriptorType> bufferTypes = {
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
			};
			int binding_count = 3;	//glsl中binding数量
			int push_constant_count = 3;	//glsl中push_constant参数数量

			pipeline_mm = new Pipeline(vkdev);
			pipeline_mm->set_optimal_local_size_xyz();
			pipeline_mm->create("matrix_multi1", specializations, bufferTypes, push_constant_count);
		}

		return 0;
	}

	void MatrixMulti_vulkan::destroy_pipeline()
	{
		delete pipeline_mm;
		pipeline_mm = nullptr;
	}

	void MatrixMulti_vulkan::forward(
		VkMat& left_blob, 
		VkMat& right_blob, 
		VkMat& top_blob,
		VkCompute& cmd,
		Option& opt,
		int m,
		int n,
		int k) {

		int elemsize = sizeof(float);
		int elempack = 1;
		top_blob.create(n, m, 1, elemsize, elempack, opt.blob_vkallocator, opt.staging_vkallocator);

		// binding in glsl
		std::vector<VkMat> bindings(3);
		bindings[0] = left_blob;
		bindings[1] = right_blob;
		bindings[2] = top_blob;

		// push_constant in glsl
		std::vector<vk_constant_type> constants(3);
		constants[0].i = m;
		constants[1].i = n;
		constants[2].i = k;

		const Pipeline* pipeline = pipeline_mm;

		cmd.record_pipeline(pipeline, bindings, constants, top_blob);
	}

}
} // namespace ncnn
