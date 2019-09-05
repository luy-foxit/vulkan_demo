#pragma once

namespace iml {
namespace train {


class VkAllocator;

class Allocator;
class Option
{
public:
    // default option
    Option();

public:

    // blob memory allocator
    Allocator* blob_allocator;

    // workspace memory allocator
    Allocator* workspace_allocator;

    // blob memory allocator
    VkAllocator* blob_vkallocator;

    // workspace memory allocator
    VkAllocator* workspace_vkallocator;

    // staging memory allocator
    VkAllocator* staging_vkallocator;

    // enable winograd convolution optimization
    // improve convolution 3x3 stride1 performace, may consume more memory
    // changes should be applied before loading network structure and weight
    // enabled by default
    bool use_winograd_convolution;

    // enable sgemm convolution optimization
    // improve convolution 1x1 stride1 performace, may consume more memory
    // changes should be applied before loading network structure and weight
    // enabled by default
    bool use_sgemm_convolution;

    // enable quantized int8 inference
    // use low-precision int8 path for quantized model
    // changes should be applied before loading network structure and weight
    // enabled by default
    bool use_int8_inference;

    // enable vulkan compute
    bool use_vulkan_compute;

    // enable options for gpu inference
    bool use_fp16_packed;
    bool use_fp16_storage;
    bool use_fp16_arithmetic;
    bool use_int8_storage;
    bool use_int8_arithmetic;

    //
    bool use_packing_layout;
};

}
} // namespace ncnn
