#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "allocator.h"

#include <vulkan/vulkan.h>
#include <opencv2/opencv.hpp>

namespace iml {
namespace train {

// the three dimension matrix, vulkan version
class VkMat
{
public:
    // empty
    VkMat();
    // vec
    VkMat(int w, size_t elemsize, VkAllocator* allocator, VkAllocator* staging_allocator);
    // image
    VkMat(int w, int h, size_t elemsize, VkAllocator* allocator, VkAllocator* staging_allocator);
    // dim
    VkMat(int w, int h, int c, size_t elemsize, VkAllocator* allocator, VkAllocator* staging_allocator);
    // packed vec
    VkMat(int w, size_t elemsize, int elempack, VkAllocator* allocator, VkAllocator* staging_allocator);
    // packed image
    VkMat(int w, int h, size_t elemsize, int elempack, VkAllocator* allocator, VkAllocator* staging_allocator);
    // packed dim
    VkMat(int w, int h, int c, size_t elemsize, int elempack, VkAllocator* allocator, VkAllocator* staging_allocator);
    // copy
    VkMat(const VkMat& m);
    // external vec
    VkMat(int w, VkBufferMemory* data, size_t offset, size_t elemsize, VkAllocator* allocator, VkAllocator* staging_allocator);
    // external image
    VkMat(int w, int h, VkBufferMemory* data, size_t offset, size_t elemsize, VkAllocator* allocator, VkAllocator* staging_allocator);
    // external dim
    VkMat(int w, int h, int c, VkBufferMemory* data, size_t offset, size_t elemsize, VkAllocator* allocator, VkAllocator* staging_allocator);
    // external packed vec
    VkMat(int w, VkBufferMemory* data, size_t offset, size_t elemsize, int elempack, VkAllocator* allocator, VkAllocator* staging_allocator);
    // external packed image
    VkMat(int w, int h, VkBufferMemory* data, size_t offset, size_t elemsize, int elempack, VkAllocator* allocator, VkAllocator* staging_allocator);
    // external packed dim
    VkMat(int w, int h, int c, VkBufferMemory* data, size_t offset, size_t elemsize, int elempack, VkAllocator* allocator, VkAllocator* staging_allocator);
    // release
    ~VkMat();
    // assign
    VkMat& operator=(const VkMat& m);
    // allocate vec
    void create(int w, size_t elemsize, VkAllocator* allocator, VkAllocator* staging_allocator);
    // allocate image
    void create(int w, int h, size_t elemsize, VkAllocator* allocator, VkAllocator* staging_allocator);
    // allocate dim
    void create(int w, int h, int c, size_t elemsize, VkAllocator* allocator, VkAllocator* staging_allocator);
    // allocate packed vec
    void create(int w, size_t elemsize, int elempack, VkAllocator* allocator, VkAllocator* staging_allocator);
    // allocate packed image
    void create(int w, int h, size_t elemsize, int elempack, VkAllocator* allocator, VkAllocator* staging_allocator);
    // allocate packed dim
    void create(int w, int h, int c, size_t elemsize, int elempack, VkAllocator* allocator, VkAllocator* staging_allocator);
    // allocate like
    void create_like(const cv::Mat& m, VkAllocator* allocator, VkAllocator* staging_allocator);
    // allocate like
    void create_like(const VkMat& m, VkAllocator* allocator, VkAllocator* staging_allocator);

    // staging buffer
    void prepare_staging_buffer();
    void discard_staging_buffer();

    // copy
    void upload(const cv::Mat& m);
    void download(cv::Mat& m) const;
	void download(std::vector<float>& m) const;

    // mapped
    //cv::Mat mapped() const;
    void* mapped_ptr() const;

    // refcount++
    void addref();
    // refcount--
    void release();

    bool empty() const;
    size_t total() const;

    // data reference
    VkMat channel(int c);
    const VkMat channel(int c) const;

    // range reference
    VkMat channel_range(int c, int channels);
    const VkMat channel_range(int c, int channels) const;
    VkMat row_range(int y, int rows);
    const VkMat row_range(int y, int rows) const;
    VkMat range(int x, int n);
    const VkMat range(int x, int n) const;

    // low-level reference
    VkBuffer buffer() const;
    size_t buffer_offset() const;
    VkBuffer staging_buffer() const;
    size_t staging_buffer_offset() const;

    // device buffer
    VkBufferMemory* data;
    // subrange offset
    size_t offset;

    // staging buffer
    VkBufferMemory* staging_data;

    // pointer to the reference counter
    // when points to user-allocated data, the pointer is NULL
    int* refcount;
    int* staging_refcount;

    // element size in bytes
    // 4 = float32/int32
    // 2 = float16
    // 1 = int8/uint8
    // 0 = empty
    size_t elemsize;

    // packed count inside element
    // c/1-h-w-1  h/1-w-1  w/1-1  scalar
    // c/4-h-w-4  h/4-w-4  w/4-4  sse/neon
    // c/8-h-w-8  h/8-w-8  w/8-8  avx/fp16
    int elempack;

    // the allocator
    VkAllocator* allocator;
    VkAllocator* staging_allocator;

    // the dimensionality
    int dims;

    int w;
    int h;
    int c;

    size_t cstep;
};

inline VkMat::VkMat()
    : data(0), offset(0), staging_data(0), refcount(0), staging_refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
}

inline VkMat::VkMat(int _w, size_t _elemsize, VkAllocator* _allocator, VkAllocator* _staging_allocator)
    : data(0), offset(0), staging_data(0), refcount(0), staging_refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _elemsize, _allocator, _staging_allocator);
}

inline VkMat::VkMat(int _w, int _h, size_t _elemsize, VkAllocator* _allocator, VkAllocator* _staging_allocator)
    : data(0), offset(0), staging_data(0), refcount(0), staging_refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _elemsize, _allocator, _staging_allocator);
}

inline VkMat::VkMat(int _w, int _h, int _c, size_t _elemsize, VkAllocator* _allocator, VkAllocator* _staging_allocator)
    : data(0), offset(0), staging_data(0), refcount(0), staging_refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _c, _elemsize, _allocator, _staging_allocator);
}

inline VkMat::VkMat(int _w, size_t _elemsize, int _elempack, VkAllocator* _allocator, VkAllocator* _staging_allocator)
    : data(0), offset(0), staging_data(0), refcount(0), staging_refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _elemsize, _elempack, _allocator, _staging_allocator);
}

inline VkMat::VkMat(int _w, int _h, size_t _elemsize, int _elempack, VkAllocator* _allocator, VkAllocator* _staging_allocator)
    : data(0), offset(0), staging_data(0), refcount(0), staging_refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _elemsize, _elempack, _allocator, _staging_allocator);
}

inline VkMat::VkMat(int _w, int _h, int _c, size_t _elemsize, int _elempack, VkAllocator* _allocator, VkAllocator* _staging_allocator)
    : data(0), offset(0), staging_data(0), refcount(0), staging_refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _c, _elemsize, _elempack, _allocator, _staging_allocator);
}

inline VkMat::VkMat(const VkMat& m)
    : data(m.data), offset(m.offset), staging_data(m.staging_data), refcount(m.refcount), staging_refcount(m.staging_refcount), elemsize(m.elemsize), elempack(m.elempack), allocator(m.allocator), staging_allocator(m.staging_allocator), dims(m.dims), w(m.w), h(m.h), c(m.c)
{
    if (refcount)
        NCNN_XADD(refcount, 1);

    if (staging_refcount)
        NCNN_XADD(staging_refcount, 1);

    cstep = m.cstep;
}

inline VkMat::VkMat(int _w, VkBufferMemory* _data, size_t _offset, size_t _elemsize, VkAllocator* _allocator, VkAllocator* _staging_allocator)
    : data(_data), offset(_offset), staging_data(0), refcount(0), staging_refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), staging_allocator(_staging_allocator), dims(1), w(_w), h(1), c(1)
{
    cstep = w;
}

inline VkMat::VkMat(int _w, int _h, VkBufferMemory* _data, size_t _offset, size_t _elemsize, VkAllocator* _allocator, VkAllocator* _staging_allocator)
    : data(_data), offset(_offset), staging_data(0), refcount(0), staging_refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), staging_allocator(_staging_allocator), dims(2), w(_w), h(_h), c(1)
{
    cstep = w * h;
}

inline VkMat::VkMat(int _w, int _h, int _c, VkBufferMemory* _data, size_t _offset, size_t _elemsize, VkAllocator* _allocator, VkAllocator* _staging_allocator)
    : data(_data), offset(_offset), staging_data(0), refcount(0), staging_refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), staging_allocator(_staging_allocator), dims(3), w(_w), h(_h), c(_c)
{
    cstep = alignSize(w * h * elemsize, 16) / elemsize;
}

inline VkMat::VkMat(int _w, VkBufferMemory* _data, size_t _offset, size_t _elemsize, int _elempack, VkAllocator* _allocator, VkAllocator* _staging_allocator)
    : data(_data), offset(_offset), staging_data(0), refcount(0), staging_refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), staging_allocator(_staging_allocator), dims(1), w(_w), h(1), c(1)
{
    cstep = w;
}

inline VkMat::VkMat(int _w, int _h, VkBufferMemory* _data, size_t _offset, size_t _elemsize, int _elempack, VkAllocator* _allocator, VkAllocator* _staging_allocator)
    : data(_data), offset(_offset), staging_data(0), refcount(0), staging_refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), staging_allocator(_staging_allocator), dims(2), w(_w), h(_h), c(1)
{
    cstep = w * h;
}

inline VkMat::VkMat(int _w, int _h, int _c, VkBufferMemory* _data, size_t _offset, size_t _elemsize, int _elempack, VkAllocator* _allocator, VkAllocator* _staging_allocator)
    : data(_data), offset(_offset), staging_data(0), refcount(0), staging_refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), staging_allocator(_staging_allocator), dims(3), w(_w), h(_h), c(_c)
{
    cstep = alignSize(w * h * elemsize, 16) / elemsize;
}

inline VkMat::~VkMat()
{
    release();
}

inline VkMat& VkMat::operator=(const VkMat& m)
{
    if (this == &m)
        return *this;

    if (m.refcount)
        NCNN_XADD(m.refcount, 1);

    if (m.staging_refcount)
        NCNN_XADD(m.staging_refcount, 1);

    release();

    data = m.data;
    offset = m.offset;
    staging_data = m.staging_data;
    refcount = m.refcount;
    staging_refcount = m.staging_refcount;
    elemsize = m.elemsize;
    elempack = m.elempack;
    allocator = m.allocator;
    staging_allocator = m.staging_allocator;

    dims = m.dims;
    w = m.w;
    h = m.h;
    c = m.c;

    cstep = m.cstep;

    return *this;
}

inline void VkMat::create(int _w, size_t _elemsize, VkAllocator* _allocator, VkAllocator* _staging_allocator)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == 1 && allocator == _allocator && staging_allocator == _staging_allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;
    staging_allocator = _staging_allocator;

    dims = 1;
    w = _w;
    h = 1;
    c = 1;

    cstep = w;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);

        data = allocator->fastMalloc(totalsize);
        offset = 0;

        refcount = (int*)((unsigned char*)data + offsetof(VkBufferMemory, refcount));
        *refcount = 1;
    }
}

inline void VkMat::create(int _w, int _h, size_t _elemsize, VkAllocator* _allocator, VkAllocator* _staging_allocator)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && elempack == 1 && allocator == _allocator && staging_allocator == _staging_allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;
    staging_allocator = _staging_allocator;

    dims = 2;
    w = _w;
    h = _h;
    c = 1;

    cstep = w * h;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);

        data = allocator->fastMalloc(totalsize);
        offset = 0;

        refcount = (int*)((unsigned char*)data + offsetof(VkBufferMemory, refcount));
        *refcount = 1;
    }
}

inline void VkMat::create(int _w, int _h, int _c, size_t _elemsize, VkAllocator* _allocator, VkAllocator* _staging_allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == 1 && allocator == _allocator && staging_allocator == _staging_allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;
    staging_allocator = _staging_allocator;

    dims = 3;
    w = _w;
    h = _h;
    c = _c;

    cstep = alignSize(w * h * elemsize, 16) / elemsize;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);

        data = allocator->fastMalloc(totalsize);
        offset = 0;

        refcount = (int*)((unsigned char*)data + offsetof(VkBufferMemory, refcount));
        *refcount = 1;
    }
}

inline void VkMat::create(int _w, size_t _elemsize, int _elempack, VkAllocator* _allocator, VkAllocator* _staging_allocator)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == _elempack && allocator == _allocator && staging_allocator == _staging_allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;
    staging_allocator = _staging_allocator;

    dims = 1;
    w = _w;
    h = 1;
    c = 1;

    cstep = w;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);

        data = allocator->fastMalloc(totalsize);
        offset = 0;

        refcount = (int*)((unsigned char*)data + offsetof(VkBufferMemory, refcount));
        *refcount = 1;
    }
}

inline void VkMat::create(int _w, int _h, size_t _elemsize, int _elempack, VkAllocator* _allocator, VkAllocator* _staging_allocator)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && elempack == _elempack && allocator == _allocator && staging_allocator == _staging_allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;
    staging_allocator = _staging_allocator;

    dims = 2;
    w = _w;
    h = _h;
    c = 1;

    cstep = w * h;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);

        data = allocator->fastMalloc(totalsize);
        offset = 0;

        refcount = (int*)((unsigned char*)data + offsetof(VkBufferMemory, refcount));
        *refcount = 1;
    }
}

inline void VkMat::create(int _w, int _h, int _c, size_t _elemsize, int _elempack, VkAllocator* _allocator, VkAllocator* _staging_allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == _elempack && allocator == _allocator && staging_allocator == _staging_allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;
    staging_allocator = _staging_allocator;

    dims = 3;
    w = _w;
    h = _h;
    c = _c;

    cstep = alignSize(w * h * elemsize, 16) / elemsize;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);

        data = allocator->fastMalloc(totalsize);
        offset = 0;

        refcount = (int*)((unsigned char*)data + offsetof(VkBufferMemory, refcount));
        *refcount = 1;
    }
}

inline void VkMat::create_like(const cv::Mat& m, VkAllocator* _allocator, VkAllocator* _staging_allocator)
{
	int elempack = 1;
#if 0
    if (m.dims == 1)
        create(m.cols, m.elemSize(), elempack, _allocator, _staging_allocator);
    else if (m.dims == 2)
        create(m.cols, m.rows, m.elemSize(), elempack, _allocator, _staging_allocator);
    else if (m.dims == 3)
#endif
        create(m.cols, m.rows, m.channels(), m.elemSize1(), elempack, _allocator, _staging_allocator);
}

inline void VkMat::create_like(const VkMat& m, VkAllocator* _allocator, VkAllocator* _staging_allocator)
{
    if (m.dims == 1)
        create(m.w, m.elemsize, m.elempack, _allocator, _staging_allocator);
    else if (m.dims == 2)
        create(m.w, m.h, m.elemsize, m.elempack, _allocator, _staging_allocator);
    else if (m.dims == 3)
        create(m.w, m.h, m.c, m.elemsize, m.elempack, _allocator, _staging_allocator);
}

inline void VkMat::prepare_staging_buffer()
{
    if (allocator->mappable)
        return;

    if (staging_allocator && staging_data)
        return;

    size_t totalsize = alignSize(total() * elemsize, 4);
    staging_data = staging_allocator->fastMalloc(totalsize);

    staging_refcount = (int*)((unsigned char*)staging_data + offsetof(VkBufferMemory, refcount));
    *staging_refcount = 1;
}

inline void VkMat::discard_staging_buffer()
{
    if (allocator->mappable)
        return;

    if (staging_refcount && NCNN_XADD(staging_refcount, -1) == 1)
    {
        if (staging_allocator && staging_data)
        {
            staging_allocator->fastFree(staging_data);
        }
    }

    staging_data = 0;
    staging_refcount = 0;
}

inline void VkMat::upload(const cv::Mat& m)
{
    memcpy(mapped_ptr(), m.data, m.total() * m.elemSize());
}

inline void VkMat::download(cv::Mat& m) const
{
    memcpy(m.data, mapped_ptr(), total() * elemsize);
}

inline void VkMat::download(std::vector<float>& m) const
{
	m.resize(total());
	memcpy(&m[0], mapped_ptr(), total() * elemsize);
}

inline void* VkMat::mapped_ptr() const
{
    VkBufferMemory* mappable_data = allocator->mappable ? data : staging_data;
    return (unsigned char*)mappable_data->mapped_ptr + mappable_data->offset + offset;
}

inline void VkMat::addref()
{
    if (refcount)
        NCNN_XADD(refcount, 1);

    if (staging_refcount)
        NCNN_XADD(staging_refcount, 1);
}

inline void VkMat::release()
{
    if (refcount && NCNN_XADD(refcount, -1) == 1)
    {
        if (allocator && data)
        {
            allocator->fastFree(data);
        }
    }

    if (staging_refcount && NCNN_XADD(staging_refcount, -1) == 1)
    {
        if (staging_allocator && staging_data)
        {
            staging_allocator->fastFree(staging_data);
        }
    }

    data = 0;
    offset = 0;
    staging_data = 0;

    elemsize = 0;
    elempack = 0;

    dims = 0;
    w = 0;
    h = 0;
    c = 0;

    cstep = 0;

    refcount = 0;
    staging_refcount = 0;
}

inline bool VkMat::empty() const
{
    return data == 0 || total() == 0;
}

inline size_t VkMat::total() const
{
    return cstep * c;
}

inline VkMat VkMat::channel(int _c)
{
    return VkMat(w, h, data, cstep * _c * elemsize, elemsize, elempack, allocator, staging_allocator);
}

inline const VkMat VkMat::channel(int _c) const
{
    return VkMat(w, h, data, cstep * _c * elemsize, elemsize, elempack, allocator, staging_allocator);
}

inline VkMat VkMat::channel_range(int _c, int channels)
{
    return VkMat(w, h, channels, data, cstep * _c * elemsize, elemsize, elempack, allocator, staging_allocator);
}

inline const VkMat VkMat::channel_range(int _c, int channels) const
{
    return VkMat(w, h, channels, data, cstep * _c * elemsize, elemsize, elempack, allocator, staging_allocator);
}

inline VkMat VkMat::row_range(int y, int rows)
{
    return VkMat(w, rows, data, w * y * elemsize, elemsize, elempack, allocator, staging_allocator);
}

inline const VkMat VkMat::row_range(int y, int rows) const
{
    return VkMat(w, rows, data, w * y * elemsize, elemsize, elempack, allocator, staging_allocator);
}

inline VkMat VkMat::range(int x, int n)
{
    return VkMat(n, data, x * elemsize, elemsize, elempack, allocator, staging_allocator);
}

inline const VkMat VkMat::range(int x, int n) const
{
    return VkMat(n, data, x * elemsize, elemsize, elempack, allocator, staging_allocator);
}

inline VkBuffer VkMat::buffer() const
{
    return data->buffer;
}

inline size_t VkMat::buffer_offset() const
{
    return data->offset + offset;
}

inline VkBuffer VkMat::staging_buffer() const
{
    return staging_data->buffer;
}

inline size_t VkMat::staging_buffer_offset() const
{
    return staging_data->offset;
}

}
} // namespace ncnn
