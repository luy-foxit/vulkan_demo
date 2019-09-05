#include "option.h"

namespace iml {
namespace train {

Option::Option()
{
    blob_allocator = 0;
    workspace_allocator = 0;

    blob_vkallocator = 0;
    workspace_vkallocator = 0;
    staging_vkallocator = 0;

    use_winograd_convolution = true;
    use_sgemm_convolution = true;
    use_int8_inference = true;
    use_vulkan_compute = true;

    use_fp16_packed = true;
    use_fp16_storage = true;
    use_fp16_arithmetic = false;
    use_int8_storage = true;
    use_int8_arithmetic = false;

    use_packing_layout = false;
}

}
} // namespace ncnn
