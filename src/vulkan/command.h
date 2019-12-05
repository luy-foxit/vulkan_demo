// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
#pragma once


#include <vector>
#include <vulkan/vulkan.h>
#include "pipeline.h"
#include "vulkan_device.h"
#include "vkmat.h"

namespace iml {
namespace train {

	class Command
	{
	public:
		Command(const VulkanDevice* vkdev, uint32_t queue_family_index);
		virtual ~Command();
		
		int create_command_pool();
		int create_command_buffer();

		// record issue
		int begin_command_buffer();
		int end_command_buffer();

		int queue_submit_and_wait_fence();

	protected:
		const VulkanDevice* vkdev;
		uint32_t queue_family_index;

		VkCommandPool command_pool;
		VkCommandBuffer command_buffer;
		VkFence fence;
	};

	// 计算命令
	class VkCompute : public Command
	{
	public:
		VkCompute(const VulkanDevice* vkdev);
		~VkCompute();

		void record_upload(const VkMat& m);

		void record_download(const VkMat& m);

		void record_clone(const VkMat& src, const VkMat& dst);

		void record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& bindings, const std::vector<vk_constant_type>& constants, const VkMat& m);
		int submit_and_wait();
		int reset();

	protected:
		void copy_buffer(VkBuffer src, size_t src_offset, VkBuffer dst, size_t dst_offset, size_t size);

		void compute_transfer_barrier(VkBuffer buffer, size_t offset, size_t size);
		void transfer_compute_barrier(VkBuffer buffer, size_t offset, size_t size);
		void compute_compute_barrier(VkBuffer buffer, size_t offset, size_t size);
		void transfer_transfer_barrier(VkBuffer buffer, size_t offset, size_t size);

		void record_prepare_compute_barrier(const VkMat& m);
		void record_transfer_compute_barrier(const VkMat& m);
		void record_compute_transfer_barrier(const VkMat& m);
		void record_compute_compute_barrier(const VkMat& m);
		void record_transfer_transfer_barrier(const VkMat& m);

		void record_prepare_transfer_barrier(const VkMat& m);

		void record_bind_pipeline(VkPipeline pipeline);
		void record_update_bindings(VkPipelineLayout pipeline_layout, VkDescriptorSetLayout descriptorset_layout, VkDescriptorUpdateTemplateKHR descriptor_update_template, const std::vector<VkMat>& bindings);
		void record_push_constants(VkPipelineLayout pipeline_layout, const std::vector<vk_constant_type>& constants);
		void record_dispatch(const uint32_t* group_count_xyz);

		void bind_pipeline(VkPipeline pipeline);
		void update_bindings(VkPipelineLayout pipeline_layout, VkDescriptorUpdateTemplateKHR descriptor_update_template, const std::vector<VkDescriptorBufferInfo>& descriptorBufferInfos);
		void push_constants(VkPipelineLayout pipeline_layout, const std::vector<vk_constant_type>& constants);
		void dispatch(const uint32_t* group_count_xyz);

	protected:
		// the good-old path for device without VK_KHR_push_descriptor
		std::vector<VkDescriptorPool> descriptor_pools;
		std::vector<VkDescriptorSet> descriptorsets;

		struct record_type
		{
			// 0=copy
			// 1=copy regions
			// 2=bind pipeline
			// 3=bind descriptorset
			// 4=push constants
			// 5=dispatch
			// 6=transfer-compute barrier
			// 7=compute-transfer barrier
			// 8=compute-compute barrier
			// 9=transfer-transfer barrier
			// 10=write timestamp
			int type;

			union
			{
				struct { VkBuffer src; size_t src_offset; VkBuffer dst; size_t dst_offset; size_t size; } copy;
				struct { VkBuffer src; VkBuffer dst; } copy_regions;
				struct { VkPipeline pipeline; } bind_pipeline;
				struct { VkPipelineLayout pipeline_layout; VkDescriptorSet descriptorset; } bind_descriptorset;
				struct { VkPipelineLayout pipeline_layout; } push_constants;
				struct { uint32_t group_count_xyz[3]; } dispatch;
				struct { VkBuffer buffer; size_t offset; size_t size; } transfer_compute_barrier;
				struct { VkBuffer buffer; size_t offset; size_t size; } compute_transfer_barrier;
				struct { VkBuffer buffer; size_t offset; size_t size; } compute_compute_barrier;
				struct { VkBuffer buffer; size_t offset; size_t size; } transfer_transfer_barrier;
			};

			std::vector<VkBufferCopy> regions;
			std::vector<vk_constant_type> constants;
		};
		std::vector<record_type> delayed_records;
	};

	// 数据传输命令
	class VkTransfer : public Command
	{
	public:
		VkTransfer(const VulkanDevice* vkdev);
		~VkTransfer();

		void record_upload(std::vector<float>& src, VkMat& dst);
		int submit_and_wait();

	protected:
		void copy_buffer(VkBuffer src, size_t src_offset, VkBuffer dst, size_t dst_offset, size_t size);
		void copy_buffer_regions(VkBuffer src, VkBuffer dst, const std::vector<VkBufferCopy>& regions);

	public:
		VkAllocator* weight_vkallocator;
		VkAllocator* staging_vkallocator;

	protected:
		size_t buffer_offset_alignment;
		VkBufferMemory* staging_data;

		struct record_type
		{
			size_t size;
			std::vector<float>	cpu_data;
			VkMat vkmat;
		};
		std::vector<record_type> delayed_records;
	};

}
} // namespace ncnn

