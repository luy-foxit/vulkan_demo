#pragma once

#include <vector>
#include <list>
#include <iostream>
#include "vulkan_device.h"

namespace iml {
namespace train {

	class VkBufferMemory
	{
	public:
		VkBuffer buffer;

		// the base offset assigned by allocator
		size_t offset;
		size_t capacity;

		VkDeviceMemory memory;
		void* mapped_ptr;

		// buffer state, modified by command functions internally
		// 0=null
		// 1=created
		// 2=transfer
		// 3=compute
		// 4=readonly
		mutable int state;

		// initialize and modified by mat
		int refcount;
	};

	class VkAllocator
	{
	public:
		VkAllocator(const VulkanDevice* _vkdev);
		virtual ~VkAllocator() { clear(); }
		virtual void clear() {}
		virtual VkBufferMemory* fastMalloc(size_t size) = 0;
		virtual void fastFree(VkBufferMemory* ptr) = 0;

	public:
		const VulkanDevice* vkdev;
		bool mappable;

	protected:
		VkBuffer create_buffer(size_t size, VkBufferUsageFlags usage);
		VkDeviceMemory allocate_memory(size_t size, uint32_t memory_type_index);
		VkDeviceMemory allocate_dedicated_memory(size_t size, uint32_t memory_type_index, VkBuffer buffer);
	};


	class VkBlobBufferAllocator : public VkAllocator
	{
	public:
		VkBlobBufferAllocator(const VulkanDevice* vkdev);
		virtual ~VkBlobBufferAllocator();

	public:
		// buffer block size, default=16M
		void set_block_size(size_t size);

		// release all budgets immediately
		virtual void clear();

		virtual VkBufferMemory* fastMalloc(size_t size);
		virtual void fastFree(VkBufferMemory* ptr);

	private:
		size_t block_size;
		size_t buffer_offset_alignment;
		std::vector< std::list< std::pair<size_t, size_t> > > budgets;
		std::vector<VkBufferMemory*> buffer_blocks;
	};


	class VkStagingBufferAllocator : public VkAllocator
	{
	public:
		VkStagingBufferAllocator(const VulkanDevice* vkdev);
		virtual ~VkStagingBufferAllocator();

	public:
		// ratio range 0 ~ 1
		// default cr = 0.75
		void set_size_compare_ratio(float scr);

		// release all budgets immediately
		virtual void clear();

		virtual VkBufferMemory* fastMalloc(size_t size);
		virtual void fastFree(VkBufferMemory* ptr);

	private:
		uint32_t memory_type_index;
		unsigned int size_compare_ratio;// 0~256
		std::list<VkBufferMemory*> budgets;
	};


}
}
