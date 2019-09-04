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

namespace iml {
namespace train {

	class Command
	{
	public:
		Command(const VulkanDevice* vkdev, uint32_t queue_family_index);
		virtual ~Command();
		
	protected:
		const VulkanDevice* vkdev;
		uint32_t queue_family_index;
	};

	// ��������
	class VkCompute : public Command
	{
	public:
		VkCompute(const VulkanDevice* vkdev);
		~VkCompute();
	};

	// ���ݴ�������
	class VkTransfer : public Command
	{
	public:
		VkTransfer(const VulkanDevice* vkdev);
		~VkTransfer();
	};

}
} // namespace ncnn

