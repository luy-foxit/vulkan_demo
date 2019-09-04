#include "allocator.h"

#include <stdio.h>
#include <algorithm>
#include "gpu.h"

namespace iml {
namespace train {

	static inline size_t alignSize(size_t sz, int n)
	{
		return (sz + n - 1) & -n;
	}

	static inline size_t least_common_multiple(size_t a, size_t b)
	{
		if (a == b)
			return a;

		if (a > b)
			return least_common_multiple(b, a);

		size_t lcm = b;
		while (lcm % a != 0)
		{
			lcm += b;
		}

		return lcm;
	}

	VkAllocator::VkAllocator(const VulkanDevice* _vkdev) : vkdev(_vkdev)
	{
		mappable = false;
	}

	VkBuffer VkAllocator::create_buffer(size_t size, VkBufferUsageFlags usage)
	{
		VkBufferCreateInfo bufferCreateInfo;
		bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferCreateInfo.pNext = 0;
		bufferCreateInfo.flags = 0;
		bufferCreateInfo.size = size;
		bufferCreateInfo.usage = usage;
		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		bufferCreateInfo.queueFamilyIndexCount = 0;
		bufferCreateInfo.pQueueFamilyIndices = 0;

		VkBuffer buffer;
		VkResult ret = vkCreateBuffer(vkdev->vkdevice(), &bufferCreateInfo, 0, &buffer);
		if (ret != VK_SUCCESS)
		{
			fprintf(stderr, "vkCreateBuffer failed %d\n", ret);
			return 0;
		}

		return buffer;
	}

	VkDeviceMemory VkAllocator::allocate_memory(size_t size, uint32_t memory_type_index)
	{
		VkMemoryAllocateInfo memoryAllocateInfo;
		memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memoryAllocateInfo.pNext = 0;
		memoryAllocateInfo.allocationSize = size;
		memoryAllocateInfo.memoryTypeIndex = memory_type_index;

		VkDeviceMemory memory = 0;
		VkResult ret = vkAllocateMemory(vkdev->vkdevice(), &memoryAllocateInfo, 0, &memory);
		if (ret != VK_SUCCESS)
		{
			fprintf(stderr, "vkAllocateMemory failed %d\n", ret);
		}

		return memory;
	}


	VkDeviceMemory VkAllocator::allocate_dedicated_memory(size_t size, uint32_t memory_type_index, VkBuffer buffer)
	{
		VkMemoryAllocateInfo memoryAllocateInfo;
		memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memoryAllocateInfo.pNext = 0;
		memoryAllocateInfo.allocationSize = size;
		memoryAllocateInfo.memoryTypeIndex = memory_type_index;

		VkMemoryDedicatedAllocateInfoKHR memoryDedicatedAllocateInfo;
		memoryDedicatedAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR;
		memoryDedicatedAllocateInfo.pNext = 0;
		memoryDedicatedAllocateInfo.image = 0;
		memoryDedicatedAllocateInfo.buffer = buffer;
		memoryAllocateInfo.pNext = &memoryDedicatedAllocateInfo;

		VkDeviceMemory memory = 0;
		VkResult ret = vkAllocateMemory(vkdev->vkdevice(), &memoryAllocateInfo, 0, &memory);
		if (ret != VK_SUCCESS)
		{
			fprintf(stderr, "vkAllocateMemory failed %d\n", ret);
		}

		return memory;
	}


	VkBlobBufferAllocator::VkBlobBufferAllocator(const VulkanDevice* _vkdev) : VkAllocator(_vkdev)
	{
		mappable = vkdev->info.device_local_memory_index == vkdev->info.unified_memory_index;

		buffer_offset_alignment = vkdev->info.buffer_offset_alignment;

		if (mappable)
		{
			// least common multiple for memory_map_alignment and buffer_offset_alignment
			size_t memory_map_alignment = vkdev->info.memory_map_alignment;
			buffer_offset_alignment = least_common_multiple(buffer_offset_alignment, memory_map_alignment);
		}

		block_size = alignSize(16 * 1024 * 1024, buffer_offset_alignment);// 16M
	}

	VkBlobBufferAllocator::~VkBlobBufferAllocator()
	{
		clear();
	}

	void VkBlobBufferAllocator::set_block_size(size_t _block_size)
	{
		block_size = _block_size;
	}

	void VkBlobBufferAllocator::clear()
	{
		//     fprintf(stderr, "VkBlobBufferAllocator %lu\n", buffer_blocks.size());

		for (size_t i = 0; i < buffer_blocks.size(); i++)
		{
			VkBufferMemory* ptr = buffer_blocks[i];

			//         std::list< std::pair<size_t, size_t> >::iterator it = budgets[i].begin();
			//         while (it != budgets[i].end())
			//         {
			//             fprintf(stderr, "VkBlobBufferAllocator budget %p %lu %lu\n", ptr->buffer, it->first, it->second);
			//             it++;
			//         }

			if (mappable)
				vkUnmapMemory(vkdev->vkdevice(), ptr->memory);

			vkDestroyBuffer(vkdev->vkdevice(), ptr->buffer, 0);
			vkFreeMemory(vkdev->vkdevice(), ptr->memory, 0);

			delete ptr;
		}
		buffer_blocks.clear();

		budgets.clear();
	}

	VkBufferMemory* VkBlobBufferAllocator::fastMalloc(size_t size)
	{
		size_t aligned_size = alignSize(size, buffer_offset_alignment);

		const int buffer_block_count = buffer_blocks.size();

		// find first spare space in buffer_blocks
		for (int i = 0; i < buffer_block_count; i++)
		{
			std::list< std::pair<size_t, size_t> >::iterator it = budgets[i].begin();
			while (it != budgets[i].end())
			{
				size_t budget_size = it->second;
				if (budget_size < aligned_size)
				{
					it++;
					continue;
				}

				// return sub buffer
				VkBufferMemory* ptr = new VkBufferMemory;

				ptr->buffer = buffer_blocks[i]->buffer;
				ptr->offset = it->first;
				ptr->memory = buffer_blocks[i]->memory;
				ptr->capacity = aligned_size;
				ptr->mapped_ptr = buffer_blocks[i]->mapped_ptr;
				ptr->state = 1;

				// adjust budgets
				if (budget_size == aligned_size)
				{
					budgets[i].erase(it);
				}
				else
				{
					it->first += aligned_size;
					it->second -= aligned_size;
				}

				//             fprintf(stderr, "VkBlobBufferAllocator M %p +%lu %lu\n", ptr->buffer, ptr->offset, ptr->capacity);

				return ptr;
			}
		}

		size_t new_block_size = std::max(block_size, aligned_size);

		// create new block
		VkBufferMemory* block = new VkBufferMemory;

		block->buffer = create_buffer(new_block_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
		block->offset = 0;

		// TODO respect VK_KHR_dedicated_allocation ?

		VkMemoryRequirements memoryRequirements;
		vkGetBufferMemoryRequirements(vkdev->vkdevice(), block->buffer, &memoryRequirements);

		block->memory = allocate_memory(memoryRequirements.size, vkdev->info.device_local_memory_index);

		vkBindBufferMemory(vkdev->vkdevice(), block->buffer, block->memory, 0);

		block->mapped_ptr = 0;
		if (mappable)
		{
			vkMapMemory(vkdev->vkdevice(), block->memory, 0, new_block_size, 0, &block->mapped_ptr);
		}

		buffer_blocks.push_back(block);

		// return sub buffer
		VkBufferMemory* ptr = new VkBufferMemory;

		ptr->buffer = block->buffer;
		ptr->offset = 0;
		ptr->memory = block->memory;
		ptr->capacity = aligned_size;
		ptr->mapped_ptr = block->mapped_ptr;
		ptr->state = 1;

		// adjust budgets
		std::list< std::pair<size_t, size_t> > budget;
		if (new_block_size > aligned_size)
		{
			budget.push_back(std::make_pair(aligned_size, new_block_size - aligned_size));
		}
		budgets.push_back(budget);

		//     fprintf(stderr, "VkBlobBufferAllocator M %p +%lu %lu\n", ptr->buffer, ptr->offset, ptr->capacity);

		return ptr;
	}

	void VkBlobBufferAllocator::fastFree(VkBufferMemory* ptr)
	{
		//     fprintf(stderr, "VkBlobBufferAllocator F %p +%lu %lu\n", ptr->buffer, ptr->offset, ptr->capacity);

		const int buffer_block_count = buffer_blocks.size();

		int block_index = -1;
		for (int i = 0; i < buffer_block_count; i++)
		{
			if (buffer_blocks[i]->buffer == ptr->buffer && buffer_blocks[i]->memory == ptr->memory)
			{
				block_index = i;
				break;
			}
		}

		if (block_index == -1)
		{
			fprintf(stderr, "FATAL ERROR! unlocked VkBlobBufferAllocator get wild %p\n", ptr->buffer);

			delete ptr;

			return;
		}

		// merge
		std::list< std::pair<size_t, size_t> >::iterator it_merge_left = budgets[block_index].end();
		std::list< std::pair<size_t, size_t> >::iterator it_merge_right = budgets[block_index].end();
		std::list< std::pair<size_t, size_t> >::iterator it = budgets[block_index].begin();
		for (; it != budgets[block_index].end(); it++)
		{
			if (it->first + it->second == ptr->offset)
			{
				it_merge_left = it;
			}
			else if (ptr->offset + ptr->capacity == it->first)
			{
				it_merge_right = it;
			}
		}

		if (it_merge_left != budgets[block_index].end() && it_merge_right != budgets[block_index].end())
		{
			it_merge_left->second = it_merge_right->first + it_merge_right->second - it_merge_left->first;
			budgets[block_index].erase(it_merge_right);
		}
		else if (it_merge_left != budgets[block_index].end())
		{
			it_merge_left->second = ptr->offset + ptr->capacity - it_merge_left->first;
		}
		else if (it_merge_right != budgets[block_index].end())
		{
			it_merge_right->second = it_merge_right->first + it_merge_right->second - ptr->offset;
			it_merge_right->first = ptr->offset;
		}
		else
		{
			if (ptr->offset == 0)
			{
				// chain leading block
				budgets[block_index].push_front(std::make_pair(ptr->offset, ptr->capacity));
			}
			else
			{
				budgets[block_index].push_back(std::make_pair(ptr->offset, ptr->capacity));
			}
		}

		delete ptr;
	}


	VkStagingBufferAllocator::VkStagingBufferAllocator(const VulkanDevice* _vkdev) : VkAllocator(_vkdev)
	{
		mappable = true;

		memory_type_index = vkdev->info.unified_memory_index;

		if (memory_type_index == -1)
			memory_type_index = vkdev->info.host_visible_memory_index;

		size_compare_ratio = 192;// 0.75f * 256
	}

	VkStagingBufferAllocator::~VkStagingBufferAllocator()
	{
		clear();
	}

	void VkStagingBufferAllocator::set_size_compare_ratio(float scr)
	{
		if (scr < 0.f || scr > 1.f)
		{
			fprintf(stderr, "invalid size compare ratio %f\n", scr);
			return;
		}

		size_compare_ratio = (unsigned int)(scr * 256);
	}

	void VkStagingBufferAllocator::clear()
	{
		//     fprintf(stderr, "VkStagingBufferAllocator %lu\n", budgets.size());

		std::list<VkBufferMemory*>::iterator it = budgets.begin();
		for (; it != budgets.end(); it++)
		{
			VkBufferMemory* ptr = *it;

			//         fprintf(stderr, "VkStagingBufferAllocator F %p\n", ptr->buffer);

			vkUnmapMemory(vkdev->vkdevice(), ptr->memory);
			vkDestroyBuffer(vkdev->vkdevice(), ptr->buffer, 0);
			vkFreeMemory(vkdev->vkdevice(), ptr->memory, 0);

			delete ptr;
		}
		budgets.clear();
	}

	VkBufferMemory* VkStagingBufferAllocator::fastMalloc(size_t size)
	{
		// find free budget
		std::list<VkBufferMemory*>::iterator it = budgets.begin();
		for (; it != budgets.end(); it++)
		{
			VkBufferMemory* ptr = *it;

			size_t capacity = ptr->capacity;

			// size_compare_ratio ~ 100%
			if (capacity >= size && ((capacity * size_compare_ratio) >> 8) <= size)
			{
				budgets.erase(it);

				//             fprintf(stderr, "VkStagingBufferAllocator M %p %lu reused %lu\n", ptr->buffer, size, capacity);

				return ptr;
			}
		}

		VkBufferMemory* ptr = new VkBufferMemory;

		ptr->buffer = create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
		ptr->offset = 0;

		VkMemoryRequirements memoryRequirements;
		vkGetBufferMemoryRequirements(vkdev->vkdevice(), ptr->buffer, &memoryRequirements);

		ptr->memory = allocate_memory(memoryRequirements.size, memory_type_index);

		vkBindBufferMemory(vkdev->vkdevice(), ptr->buffer, ptr->memory, 0);

		ptr->capacity = size;

		vkMapMemory(vkdev->vkdevice(), ptr->memory, 0, size, 0, &ptr->mapped_ptr);

		ptr->state = 1;

		//     fprintf(stderr, "VkStagingBufferAllocator M %p %lu\n", ptr->buffer, size);

		return ptr;
	}

	void VkStagingBufferAllocator::fastFree(VkBufferMemory* ptr)
	{
		//     fprintf(stderr, "VkStagingBufferAllocator F %p\n", ptr->buffer);

			// return to budgets
		budgets.push_back(ptr);
	}


}
} // namespace ncnn