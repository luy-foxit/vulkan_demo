#include "relu_100_vulkan.h"

namespace iml {
namespace train {

	ReLU_vulkan::ReLU_vulkan()
	{
		pipeline_relu = 0;
	}

	ReLU_vulkan::~ReLU_vulkan()
	{
		pipeline_relu = 0;
	}

	int ReLU_vulkan::create_pipeline(const Option& opt, const VulkanDevice* vkdev)
	{
		std::vector<vk_specialization_type> specializations(1);
		// pack1
		{
			pipeline_relu = new Pipeline(vkdev);
			pipeline_relu->set_optimal_local_size_xyz();
			pipeline_relu->create("relu_100", opt, specializations, 1, 5);
		}

		return 0;
	}

	int ReLU_vulkan::destroy_pipeline(const Option& opt)
	{
		delete pipeline_relu;
		pipeline_relu = 0;

		return 0;
	}

	int ReLU_vulkan::forward_inplace(cv::Mat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
	{
#if 0
		std::vector<VkMat> bindings(1);
		bindings[0] = bottom_top_blob;

		std::vector<vk_constant_type> constants(5);
		constants[0].i = bottom_top_blob.dims;
		constants[1].i = bottom_top_blob.w;
		constants[2].i = bottom_top_blob.h;
		constants[3].i = bottom_top_blob.c;
		constants[4].i = bottom_top_blob.cstep;

		const Pipeline* pipeline = elempack == 4 ? pipeline_relu_pack4 : pipeline_relu;

		cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);
#endif
		return 0;
	}

}
} // namespace ncnn
