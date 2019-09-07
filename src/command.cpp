#include "command.h"

#include <stdio.h>

namespace iml {
namespace train {

	Command::Command(const VulkanDevice* _vkdev, uint32_t _queue_family_index) : vkdev(_vkdev), queue_family_index(_queue_family_index)
	{
		create_command_pool();

		create_command_buffer();

		// create fence
		VkFenceCreateInfo fenceCreateInfo;
		fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceCreateInfo.pNext = 0;
		fenceCreateInfo.flags = 0;

		VkResult ret = vkCreateFence(vkdev->vkdevice(), &fenceCreateInfo, 0, &fence);
		if (ret != VK_SUCCESS)
		{
			fprintf(stderr, "vkCreateFence failed %d\n", ret);
		}
	}

	Command::~Command()
	{
		vkDestroyFence(vkdev->vkdevice(), fence, 0);

		vkFreeCommandBuffers(vkdev->vkdevice(), command_pool, 1, &command_buffer);

		vkDestroyCommandPool(vkdev->vkdevice(), command_pool, 0);

	}

	int Command::create_command_pool()
	{
		VkCommandPoolCreateInfo commandPoolCreateInfo;
		commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		commandPoolCreateInfo.pNext = 0;
		commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		commandPoolCreateInfo.queueFamilyIndex = queue_family_index;

		VkResult ret = vkCreateCommandPool(vkdev->vkdevice(), &commandPoolCreateInfo, 0, &command_pool);
		if (ret != VK_SUCCESS)
		{
			fprintf(stderr, "vkCreateCommandPool failed %d\n", ret);
			return -1;
		}

		return 0;
	}

	int Command::create_command_buffer()
	{
		VkCommandBufferAllocateInfo commandBufferAllocateInfo;
		commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		commandBufferAllocateInfo.pNext = 0;
		commandBufferAllocateInfo.commandPool = command_pool;
		commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		commandBufferAllocateInfo.commandBufferCount = 1;

		VkResult ret = vkAllocateCommandBuffers(vkdev->vkdevice(), &commandBufferAllocateInfo, &command_buffer);
		if (ret != VK_SUCCESS)
		{
			fprintf(stderr, "vkAllocateCommandBuffers failed %d\n", ret);
			return -1;
		}

		return 0;
	}


	int Command::begin_command_buffer()
	{
		//     fprintf(stderr, "==================== begin\n");

		VkCommandBufferBeginInfo commandBufferBeginInfo;
		commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		commandBufferBeginInfo.pNext = 0;
		commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		commandBufferBeginInfo.pInheritanceInfo = 0;

		VkResult ret = vkBeginCommandBuffer(command_buffer, &commandBufferBeginInfo);
		if (ret != VK_SUCCESS)
		{
			fprintf(stderr, "vkBeginCommandBuffer failed %d\n", ret);
			return -1;
		}

		return 0;
	}

	int Command::end_command_buffer()
	{
		//     fprintf(stderr, "==================== end\n");

		VkResult ret = vkEndCommandBuffer(command_buffer);
		if (ret != VK_SUCCESS)
		{
			fprintf(stderr, "vkEndCommandBuffer failed %d\n", ret);
			return -1;
		}

		return 0;
	}


	int Command::queue_submit_and_wait_fence()
	{
		// acquire queue and reclaim on return
		VkQueue queue = vkdev->acquire_queue(queue_family_index);
		if (queue == 0)
		{
			fprintf(stderr, "out of compute queue\n");
			return -1;
		}

		//     fprintf(stderr, "==================== submit\n");
		{
			VkSubmitInfo submitInfo;
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.pNext = 0;
			submitInfo.waitSemaphoreCount = 0;
			submitInfo.pWaitSemaphores = 0;
			submitInfo.pWaitDstStageMask = 0;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &command_buffer;
			submitInfo.signalSemaphoreCount = 0;
			submitInfo.pSignalSemaphores = 0;

			VkResult ret = vkQueueSubmit(queue, 1, &submitInfo, fence);
			if (ret != VK_SUCCESS)
			{
				fprintf(stderr, "vkQueueSubmit failed %d\n", ret);
				vkdev->reclaim_queue(queue_family_index, queue);
				return -1;
			}
		}

		//     fprintf(stderr, "==================== wait\n");
		{
			VkResult ret = vkWaitForFences(vkdev->vkdevice(), 1, &fence, VK_TRUE, UINT64_MAX);
			if (ret != VK_SUCCESS)
			{
				fprintf(stderr, "vkWaitForFences failed %d\n", ret);
				vkdev->reclaim_queue(queue_family_index, queue);
				return -1;
			}
		}

		vkdev->reclaim_queue(queue_family_index, queue);
		return 0;
	}


	VkCompute::VkCompute(const VulkanDevice* _vkdev) : Command(_vkdev, _vkdev->info.compute_queue_family_index)
	{
		if (vkdev->info.support_VK_KHR_push_descriptor)
		{
			begin_command_buffer();
		}
	}

	VkCompute::~VkCompute()
	{
	}


	void VkCompute::record_upload(const VkMat& m)
	{
		if (m.allocator->mappable)
			return;

		record_prepare_transfer_barrier(m);

		if (vkdev->info.support_VK_KHR_push_descriptor)
			return copy_buffer(m.staging_buffer(), 0, m.buffer(), m.buffer_offset(), m.total() * m.elemsize);

		record_type r;
		r.type = 0;
		r.copy.src = m.staging_buffer();
		r.copy.src_offset = 0;
		r.copy.dst = m.buffer();
		r.copy.dst_offset = m.buffer_offset();
		r.copy.size = m.total() * m.elemsize;
		delayed_records.push_back(r);
	}

	void VkCompute::record_download(const VkMat& m)
	{
		if (m.allocator->mappable)
			return;

		record_prepare_transfer_barrier(m);

		if (vkdev->info.support_VK_KHR_push_descriptor)
			return copy_buffer(m.buffer(), m.buffer_offset(), m.staging_buffer(), 0, m.total() * m.elemsize);

		record_type r;
		r.type = 0;
		r.copy.src = m.buffer();
		r.copy.src_offset = m.buffer_offset();
		r.copy.dst = m.staging_buffer();
		r.copy.dst_offset = 0;
		r.copy.size = m.total() * m.elemsize;
		delayed_records.push_back(r);
	}

	void VkCompute::record_clone(const VkMat& src, const VkMat& dst)
	{
		record_prepare_transfer_barrier(src);
		record_prepare_transfer_barrier(dst);

		if (vkdev->info.support_VK_KHR_push_descriptor)
			return copy_buffer(src.buffer(), src.buffer_offset(), dst.buffer(), dst.buffer_offset(), src.total() * src.elemsize);

		record_type r;
		r.type = 0;
		r.copy.src = src.buffer();
		r.copy.src_offset = src.buffer_offset();
		r.copy.dst = dst.buffer();
		r.copy.dst_offset = dst.buffer_offset();
		r.copy.size = src.total() * src.elemsize;
		delayed_records.push_back(r);
	}

	void VkCompute::copy_buffer(VkBuffer src, size_t src_offset, VkBuffer dst, size_t dst_offset, size_t size)
	{
		//     fprintf(stderr, "cmd copy %p to %p\n", src, dst);

		VkBufferCopy region;
		region.srcOffset = src_offset;
		region.dstOffset = dst_offset;
		region.size = size;

		vkCmdCopyBuffer(command_buffer, src, dst, 1, &region);
	}

	void VkCompute::record_prepare_transfer_barrier(const VkMat& m)
	{
		if (m.data->state == 2)
			return record_transfer_transfer_barrier(m);

		if (m.data->state == 3)
			return record_compute_transfer_barrier(m);

		m.data->state = 2;
	}


	void VkCompute::transfer_compute_barrier(VkBuffer buffer, size_t offset, size_t size)
	{
		//     fprintf(stderr, "cmd transfer_compute_barrier %p[+%lu] %lu\n", buffer, offset, size);

		VkBufferMemoryBarrier bufferBarrier;
		bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		bufferBarrier.pNext = 0;
		bufferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		bufferBarrier.buffer = buffer;
		bufferBarrier.offset = offset;
		bufferBarrier.size = size;

		VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
		VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

		vkCmdPipelineBarrier(command_buffer, srcStageMask, dstStageMask, 0, 0, 0, 1, &bufferBarrier, 0, 0);
	}

	void VkCompute::compute_transfer_barrier(VkBuffer buffer, size_t offset, size_t size)
	{
		//     fprintf(stderr, "cmd compute_transfer_barrier %p[+%lu] %lu\n", buffer, offset, size);

		VkBufferMemoryBarrier bufferBarrier;
		bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		bufferBarrier.pNext = 0;
		bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		bufferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		bufferBarrier.buffer = buffer;
		bufferBarrier.offset = offset;
		bufferBarrier.size = size;

		VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;

		vkCmdPipelineBarrier(command_buffer, srcStageMask, dstStageMask, 0, 0, 0, 1, &bufferBarrier, 0, 0);
	}

	void VkCompute::compute_compute_barrier(VkBuffer buffer, size_t offset, size_t size)
	{
		//     fprintf(stderr, "cmd compute_compute_barrier %p[+%lu] %lu\n", buffer, offset, size);

		VkBufferMemoryBarrier bufferBarrier;
		bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		bufferBarrier.pNext = 0;
		bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		bufferBarrier.buffer = buffer;
		bufferBarrier.offset = offset;
		bufferBarrier.size = size;

		VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

		vkCmdPipelineBarrier(command_buffer, srcStageMask, dstStageMask, 0, 0, 0, 1, &bufferBarrier, 0, 0);
	}

	void VkCompute::transfer_transfer_barrier(VkBuffer buffer, size_t offset, size_t size)
	{
		//     fprintf(stderr, "cmd transfer_transfer_barrier %p[+%lu] %lu\n", buffer, offset, size);

		VkBufferMemoryBarrier bufferBarrier;
		bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		bufferBarrier.pNext = 0;
		bufferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		bufferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		bufferBarrier.buffer = buffer;
		bufferBarrier.offset = offset;
		bufferBarrier.size = size;

		VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
		VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;

		vkCmdPipelineBarrier(command_buffer, srcStageMask, dstStageMask, 0, 0, 0, 1, &bufferBarrier, 0, 0);
	}

	void VkCompute::record_transfer_compute_barrier(const VkMat& m)
	{
		m.data->state = 3;

		if (vkdev->info.support_VK_KHR_push_descriptor)
			return transfer_compute_barrier(m.buffer(), m.buffer_offset(), m.total() * m.elemsize);

		record_type r;
		r.type = 6;
		r.transfer_compute_barrier.buffer = m.buffer();
		r.transfer_compute_barrier.offset = m.buffer_offset();
		r.transfer_compute_barrier.size = m.total() * m.elemsize;
		delayed_records.push_back(r);
	}

	void VkCompute::record_compute_transfer_barrier(const VkMat& m)
	{
		m.data->state = 2;

		if (vkdev->info.support_VK_KHR_push_descriptor)
			return compute_transfer_barrier(m.buffer(), m.buffer_offset(), m.total() * m.elemsize);

		record_type r;
		r.type = 7;
		r.compute_transfer_barrier.buffer = m.buffer();
		r.compute_transfer_barrier.offset = m.buffer_offset();
		r.compute_transfer_barrier.size = m.total() * m.elemsize;
		delayed_records.push_back(r);
	}

	void VkCompute::record_compute_compute_barrier(const VkMat& m)
	{
		m.data->state = 3;

		if (vkdev->info.support_VK_KHR_push_descriptor)
			return compute_compute_barrier(m.buffer(), m.buffer_offset(), m.total() * m.elemsize);

		record_type r;
		r.type = 8;
		r.compute_compute_barrier.buffer = m.buffer();
		r.compute_compute_barrier.offset = m.buffer_offset();
		r.compute_compute_barrier.size = m.total() * m.elemsize;
		delayed_records.push_back(r);
	}

	void VkCompute::record_transfer_transfer_barrier(const VkMat& m)
	{
		m.data->state = 2;

		if (vkdev->info.support_VK_KHR_push_descriptor)
			return transfer_transfer_barrier(m.buffer(), m.buffer_offset(), m.total() * m.elemsize);

		record_type r;
		r.type = 9;
		r.transfer_transfer_barrier.buffer = m.buffer();
		r.transfer_transfer_barrier.offset = m.buffer_offset();
		r.transfer_transfer_barrier.size = m.total() * m.elemsize;
		delayed_records.push_back(r);
	}

	void VkCompute::record_prepare_compute_barrier(const VkMat& m)
	{
		// 从 transfer 到 compute 的 barrier
		if (m.data->state == 2)
			return record_transfer_compute_barrier(m);

		// 从 compute 到 compute 的 barrier
		if (m.data->state == 3)
			return record_compute_compute_barrier(m);

		m.data->state = 3;
	}

	void VkCompute::record_bind_pipeline(VkPipeline pipeline)
	{
		if (vkdev->info.support_VK_KHR_push_descriptor)
			return bind_pipeline(pipeline);

		record_type r;
		r.type = 2;
		r.bind_pipeline.pipeline = pipeline;
		delayed_records.push_back(r);
	}


	void VkCompute::bind_pipeline(VkPipeline pipeline)
	{
		//fprintf(stderr, "cmd bind_pipeline %p\n", pipeline);

		vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	}

	void VkCompute::record_update_bindings(VkPipelineLayout pipeline_layout, VkDescriptorSetLayout descriptorset_layout, VkDescriptorUpdateTemplateKHR descriptor_update_template, const std::vector<VkMat>& bindings)
	{
		const int binding_count = bindings.size();

		if (binding_count == 0)
			return;

		std::vector<VkDescriptorBufferInfo> descriptorBufferInfos(binding_count);
		for (int i = 0; i < binding_count; i++)
		{
			descriptorBufferInfos[i].buffer = bindings[i].buffer();
			descriptorBufferInfos[i].offset = bindings[i].buffer_offset();
			descriptorBufferInfos[i].range = bindings[i].total() * bindings[i].elemsize;
		}

		if (vkdev->info.support_VK_KHR_push_descriptor)
			return update_bindings(pipeline_layout, descriptor_update_template, descriptorBufferInfos);

		// create new descriptor_pool and descriptorset
		VkDescriptorPool descriptor_pool;
		{
			VkDescriptorPoolSize poolSize;
			poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			poolSize.descriptorCount = binding_count;

			VkDescriptorPoolCreateInfo descriptorPoolCreateInfo;
			descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
			descriptorPoolCreateInfo.pNext = 0;
			descriptorPoolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
			descriptorPoolCreateInfo.maxSets = 1;
			descriptorPoolCreateInfo.poolSizeCount = 1;
			descriptorPoolCreateInfo.pPoolSizes = &poolSize;

			VkResult ret = vkCreateDescriptorPool(vkdev->vkdevice(), &descriptorPoolCreateInfo, 0, &descriptor_pool);
			if (ret != VK_SUCCESS)
			{
				fprintf(stderr, "vkCreateDescriptorPool failed %d\n", ret);
				return;
			}
		}
		descriptor_pools.push_back(descriptor_pool);

		VkDescriptorSet descriptorset;
		{
			VkDescriptorSetAllocateInfo descriptorSetAllocateInfo;
			descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			descriptorSetAllocateInfo.pNext = 0;
			descriptorSetAllocateInfo.descriptorPool = descriptor_pool;
			descriptorSetAllocateInfo.descriptorSetCount = 1;
			descriptorSetAllocateInfo.pSetLayouts = &descriptorset_layout;

			VkResult ret = vkAllocateDescriptorSets(vkdev->vkdevice(), &descriptorSetAllocateInfo, &descriptorset);
			if (ret != VK_SUCCESS)
			{
				fprintf(stderr, "vkAllocateDescriptorSets failed %d\n", ret);
				return;
			}
		}
		descriptorsets.push_back(descriptorset);

		//     fprintf(stderr, "update descriptorset %p\n", descriptorset);

		if (vkdev->info.support_VK_KHR_descriptor_update_template)
		{
			vkdev->vkUpdateDescriptorSetWithTemplateKHR(vkdev->vkdevice(), descriptorset, descriptor_update_template, descriptorBufferInfos.data());
		}
		else
		{
			std::vector<VkWriteDescriptorSet> writeDescriptorSets(binding_count);
			for (int i = 0; i < binding_count; i++)
			{
				writeDescriptorSets[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				writeDescriptorSets[i].pNext = 0;
				writeDescriptorSets[i].dstSet = descriptorset;
				writeDescriptorSets[i].dstBinding = i;
				writeDescriptorSets[i].dstArrayElement = 0;
				writeDescriptorSets[i].descriptorCount = 1;
				writeDescriptorSets[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				writeDescriptorSets[i].pImageInfo = 0;
				writeDescriptorSets[i].pBufferInfo = &descriptorBufferInfos[i];
				writeDescriptorSets[i].pTexelBufferView = 0;
			}

			vkUpdateDescriptorSets(vkdev->vkdevice(), binding_count, writeDescriptorSets.data(), 0, 0);
		}

		record_type r;
		r.type = 3;
		r.bind_descriptorset.pipeline_layout = pipeline_layout;
		r.bind_descriptorset.descriptorset = descriptorset;
		delayed_records.push_back(r);
	}

	void VkCompute::update_bindings(VkPipelineLayout pipeline_layout, VkDescriptorUpdateTemplateKHR descriptor_update_template, const std::vector<VkDescriptorBufferInfo>& descriptorBufferInfos)
	{
		//     fprintf(stderr, "cmd update_bindings %p %p\n", pipeline_layout, descriptor_update_template);

		vkdev->vkCmdPushDescriptorSetWithTemplateKHR(command_buffer, descriptor_update_template, pipeline_layout, 0, descriptorBufferInfos.data());
	}


	void VkCompute::record_push_constants(VkPipelineLayout pipeline_layout, const std::vector<vk_constant_type>& constants)
	{
		if (vkdev->info.support_VK_KHR_push_descriptor)
			return push_constants(pipeline_layout, constants);

		record_type r;
		r.type = 4;
		r.push_constants.pipeline_layout = pipeline_layout;
		r.constants = constants;
		delayed_records.push_back(r);
	}

	void VkCompute::push_constants(VkPipelineLayout pipeline_layout, const std::vector<vk_constant_type>& constants)
	{
		//     fprintf(stderr, "cmd push_constants %p\n", pipeline_layout);

		vkCmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, constants.size() * sizeof(vk_constant_type), constants.data());
	}


	void VkCompute::record_dispatch(const uint32_t* group_count_xyz)
	{
		if (vkdev->info.support_VK_KHR_push_descriptor)
			return dispatch(group_count_xyz);

		record_type r;
		r.type = 5;
		r.dispatch.group_count_xyz[0] = group_count_xyz[0];	//x轴中分配的local workgroup的数量
		r.dispatch.group_count_xyz[1] = group_count_xyz[1];	//y轴中分配的local workgroup的数量
		r.dispatch.group_count_xyz[2] = group_count_xyz[2];	//z轴中分配的local workgroup的数量。
		delayed_records.push_back(r);
	}
	void VkCompute::dispatch(const uint32_t* group_count_xyz)
	{
		//     fprintf(stderr, "cmd dispatch %d %d %d\n", group_count_xyz[0], group_count_xyz[1], group_count_xyz[2]);

		vkCmdDispatch(command_buffer, group_count_xyz[0], group_count_xyz[1], group_count_xyz[2]);
	}


	void VkCompute::record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& bindings, const std::vector<vk_constant_type>& constants, const VkMat& m)
	{
		const int binding_count = bindings.size();
		for (int i = 0; i < binding_count; i++)
		{
			// skip readonly weight blob
			if (bindings[i].data->state == 4)
				continue;

			// transfer 和 compute 的 barrier
			record_prepare_compute_barrier(bindings[i]);
		}

		// 把 pipeline 对象绑定到 command buffer
		record_bind_pipeline(pipeline->pipeline);

		// 记录图片数据 (glsl中的binding)
		record_update_bindings(pipeline->pipeline_layout, pipeline->descriptorset_layout, pipeline->descriptor_update_template, bindings);

		// 记录 const 数据(glsl中的push_constant)
		record_push_constants(pipeline->pipeline_layout, constants);

		uint32_t group_count_xyz[3];
		group_count_xyz[0] = (m.w + pipeline->local_size_x - 1) / pipeline->local_size_x;
		group_count_xyz[1] = (m.h + pipeline->local_size_y - 1) / pipeline->local_size_y;
		group_count_xyz[2] = (m.c + pipeline->local_size_z - 1) / pipeline->local_size_z;

		// 记录xyz轴workgroup数量
		record_dispatch(group_count_xyz);
	}


	int VkCompute::submit_and_wait()
	{
		if (vkdev->info.support_VK_KHR_push_descriptor)
		{
			end_command_buffer();

			return queue_submit_and_wait_fence();
		}
		
#if 0	//--TODO
		begin_command_buffer();

		// handle delayed records
		for (size_t i = 0; i < delayed_records.size(); i++)
		{
			const record_type& r = delayed_records[i];

			switch (r.type)
			{
			case 0:
				copy_buffer(r.copy.src, r.copy.src_offset, r.copy.dst, r.copy.dst_offset, r.copy.size);
				break;
			case 1:
				copy_buffer_regions(r.copy_regions.src, r.copy_regions.dst, r.regions);
				break;
			case 2:
				bind_pipeline(r.bind_pipeline.pipeline);
				break;
			case 3:
				bind_descriptorset(r.bind_descriptorset.pipeline_layout, r.bind_descriptorset.descriptorset);
				break;
			case 4:
				push_constants(r.push_constants.pipeline_layout, r.constants);
				break;
			case 5:
				dispatch(r.dispatch.group_count_xyz);
				break;
			case 6:
				transfer_compute_barrier(r.transfer_compute_barrier.buffer, r.transfer_compute_barrier.offset, r.transfer_compute_barrier.size);
				break;
			case 7:
				compute_transfer_barrier(r.compute_transfer_barrier.buffer, r.compute_transfer_barrier.offset, r.compute_transfer_barrier.size);
				break;
			case 8:
				compute_compute_barrier(r.compute_compute_barrier.buffer, r.compute_compute_barrier.offset, r.compute_compute_barrier.size);
				break;
			case 9:
				transfer_transfer_barrier(r.compute_compute_barrier.buffer, r.compute_compute_barrier.offset, r.compute_compute_barrier.size);
				break;
#if NCNN_BENCHMARK
			case 10:
				write_timestamp(r.write_timestamp.query);
				break;
#endif // NCNN_BENCHMARK
			}
		}

		end_command_buffer();

		delayed_records.clear();

		return queue_submit_and_wait_fence();
#endif
	}


	int VkCompute::reset()
	{
		//     fprintf(stderr, "cmd reset\n");

		VkResult ret = vkResetCommandBuffer(command_buffer, 0);
		if (ret != VK_SUCCESS)
		{
			fprintf(stderr, "vkResetCommandBuffer failed %d\n", ret);
			return -1;
		}

		ret = vkResetFences(vkdev->vkdevice(), 1, &fence);
		if (ret != VK_SUCCESS)
		{
			fprintf(stderr, "vkResetFences failed %d\n", ret);
			return -1;
		}

		if (vkdev->info.support_VK_KHR_push_descriptor)
		{
			begin_command_buffer();
		}

		return 0;
	}


	VkTransfer::VkTransfer(const VulkanDevice* _vkdev) : Command(_vkdev, _vkdev->info.transfer_queue_family_index)
	{
		buffer_offset_alignment = vkdev->info.buffer_offset_alignment;
		staging_data = 0;
	}

	VkTransfer::~VkTransfer()
	{
	}

	void VkTransfer::record_upload(std::vector<float>& src, VkMat& dst) {
		dst.create(src.size(), sizeof(float), weight_vkallocator, staging_vkallocator);
		
		// set weight blob as readonly
		dst.data->state = 4;
		if (dst.allocator->mappable)
		{
			//dst.upload(src);
			return;
		}

		record_type r;
		//r.size = src_flattened.total() * src_flattened.elemsize;
		r.size = src.size() * sizeof(float);
		r.cpu_data = src;
		r.vkmat = dst;
		delayed_records.push_back(r);
	}

	int VkTransfer::submit_and_wait()
	{
		if (delayed_records.empty())
			return 0;

		int transfer_count = delayed_records.size();

		// solve staging buffer size
		size_t staging_buffer_size = 0;
		for (int i = 0; i < transfer_count; i++)
		{
			const record_type& r = delayed_records[i];
			staging_buffer_size += alignSize(r.size, buffer_offset_alignment);
		}

		// allocate staging buffer
		staging_data = staging_vkallocator->fastMalloc(staging_buffer_size);

		// copy upload data
		size_t mapped_ptr_offset = 0;
		for (int i = 0; i < transfer_count; i++)
		{
			const record_type& r = delayed_records[i];

			memcpy((unsigned char*)staging_data->mapped_ptr + mapped_ptr_offset, r.cpu_data.data(), r.size);

			mapped_ptr_offset += alignSize(r.size, buffer_offset_alignment);
		}

		begin_command_buffer();

		//     fprintf(stderr, "cmd transfer %p %lu\n", staging_data->buffer, staging_buffer_size);

			// handle delayed records
		size_t staging_buffer_offset = 0;
		for (int i = 0; i < transfer_count; i++)
		{
			const record_type& r = delayed_records[i];

			copy_buffer(staging_data->buffer, staging_buffer_offset, r.vkmat.buffer(), r.vkmat.buffer_offset(), r.size);

			staging_buffer_offset += alignSize(r.size, buffer_offset_alignment);
		}

		end_command_buffer();

		int ret = queue_submit_and_wait_fence();

		// deallocate staging buffer
		staging_vkallocator->fastFree(staging_data);
		staging_data = 0;

		delayed_records.clear();

		return ret;
	}


	void VkTransfer::copy_buffer(VkBuffer src, size_t src_offset, VkBuffer dst, size_t dst_offset, size_t size)
	{
		//     fprintf(stderr, "cmd copy %p to %p\n", src, dst);

		VkBufferCopy region;
		region.srcOffset = src_offset;
		region.dstOffset = dst_offset;
		region.size = size;

		vkCmdCopyBuffer(command_buffer, src, dst, 1, &region);
	}

	void VkTransfer::copy_buffer_regions(VkBuffer src, VkBuffer dst, const std::vector<VkBufferCopy>& regions)
	{
		//     fprintf(stderr, "cmd copy regions %p to %p\n", src, dst);

		vkCmdCopyBuffer(command_buffer, src, dst, regions.size(), regions.data());
	}



}
} // namespace ncnn

