#pragma once

#include <vector>
#include <list>
#include <iostream>
#include "vulkan_device.h"

// exchange-add operation for atomic operations on reference counters
#if defined __INTEL_COMPILER && !(defined WIN32 || defined _WIN32)
// atomic increment on the linux version of the Intel(tm) compiler
#  define NCNN_XADD(addr, delta) (int)_InterlockedExchangeAdd(const_cast<void*>(reinterpret_cast<volatile void*>(addr)), delta)
#elif defined __GNUC__
#  if defined __clang__ && __clang_major__ >= 3 && !defined __ANDROID__ && !defined __EMSCRIPTEN__ && !defined(__CUDACC__)
#    ifdef __ATOMIC_ACQ_REL
#      define NCNN_XADD(addr, delta) __c11_atomic_fetch_add((_Atomic(int)*)(addr), delta, __ATOMIC_ACQ_REL)
#    else
#      define NCNN_XADD(addr, delta) __atomic_fetch_add((_Atomic(int)*)(addr), delta, 4)
#    endif
#  else
#    if defined __ATOMIC_ACQ_REL && !defined __clang__
// version for gcc >= 4.7
#      define NCNN_XADD(addr, delta) (int)__atomic_fetch_add((unsigned*)(addr), (unsigned)(delta), __ATOMIC_ACQ_REL)
#    else
#      define NCNN_XADD(addr, delta) (int)__sync_fetch_and_add((unsigned*)(addr), (unsigned)(delta))
#    endif
#  endif
#elif defined _MSC_VER && !defined RC_INVOKED
#  include <intrin.h>
#  define NCNN_XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile*)addr, delta)
#else
// thread-unsafe branch
static inline int NCNN_XADD(int* addr, int delta) { int tmp = *addr; *addr += delta; return tmp; }
#endif

namespace iml {
namespace train {

	// Aligns a buffer size to the specified number of bytes
	// The function returns the minimum number that is greater or equal to sz and is divisible by n
	// sz Buffer size to align
	// n Alignment size that must be a power of two
	static inline size_t alignSize(size_t sz, int n)
	{
		return (sz + n - 1) & -n;
	}

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

	class VkWeightBufferAllocator : public VkAllocator
	{
	public:
		VkWeightBufferAllocator(const VulkanDevice* vkdev);
		virtual ~VkWeightBufferAllocator();

	public:
		// buffer block size, default=8M
		void set_block_size(size_t block_size);

		// release all blocks immediately
		virtual void clear();

	public:
		virtual VkBufferMemory* fastMalloc(size_t size);
		virtual void fastFree(VkBufferMemory* ptr);

	private:
		size_t block_size;
		size_t buffer_offset_alignment;
		std::vector<size_t> buffer_block_free_spaces;
		std::vector<VkBufferMemory*> buffer_blocks;
		std::vector<VkBufferMemory*> dedicated_buffer_blocks;
	};

	class VkWeightStagingBufferAllocator : public VkAllocator
	{
	public:
		VkWeightStagingBufferAllocator(const VulkanDevice* vkdev);
		virtual ~VkWeightStagingBufferAllocator();

	public:
		virtual VkBufferMemory* fastMalloc(size_t size);
		virtual void fastFree(VkBufferMemory* ptr);

	private:
		uint32_t memory_type_index;
	};

}
}
