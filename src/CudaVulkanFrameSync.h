#pragma once
#include <stdexcept>
#include "cuda.h"
#include "VKRenderer.h"
#include "VkExt.h"

// CUDA/Vulkan frame handoff through one shared timeline semaphore
// instead of host synchronization, so the CPU can prepare the next frame while the GPU works.
// Iteration k: CUDA waits for 2k (Vulkan finished blitting the previous image) before touching the
// shared image and signals 2k+1 when it is complete; Vulkan waits for 2k+1 before its blit and
// signals 2k+2. Iterations without a Vulkan submit are completed from the host.
struct CudaVulkanFrameSync {
	inline static VkSemaphore timeline = VK_NULL_HANDLE;
	inline static CUexternalSemaphore cudaTimeline = nullptr;
	inline static uint64_t iteration = 0;
	inline static bool cudaSignaled = false;
	inline static bool vulkanSignaled = false;

	static void check(bool value, const char* message) {
		if (!value) throw std::runtime_error(std::string("pipelined present: ") + message);
	}

	static uint64_t cudaWaitValue()     { return 2 * iteration; }
	static uint64_t cudaSignalValue()   { return 2 * iteration + 1; }
	static uint64_t vulkanSignalValue() { return 2 * iteration + 2; }

	static void init() {
		if (timeline != VK_NULL_HANDLE) return;
#ifdef _WIN32
		VkExportSemaphoreCreateInfo exportInfo = {};
		exportInfo.sType       = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
		exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

		VkSemaphoreTypeCreateInfo typeInfo = {};
		typeInfo.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
		typeInfo.pNext         = &exportInfo;
		typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
		typeInfo.initialValue  = 0;

		VkSemaphoreCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		createInfo.pNext = &typeInfo;
		check(vkCreateSemaphore(VKRenderer::device, &createInfo, nullptr, &timeline) == VK_SUCCESS, "timeline creation failed");

		VkSemaphoreGetWin32HandleInfoKHR handleInfo = {};
		handleInfo.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
		handleInfo.semaphore  = timeline;
		handleInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
		HANDLE handle = nullptr;
		check(vkGetSemaphoreWin32HandleKHR(VKRenderer::device, &handleInfo, &handle) == VK_SUCCESS, "timeline export failed");

		CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC desc{};
		desc.type                = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32;
		desc.handle.win32.handle = handle;
		check(cuImportExternalSemaphore(&cudaTimeline, &desc) == CUDA_SUCCESS, "CUDA timeline import failed");
		CloseHandle(handle); // CUDA holds its own reference
#else
		check(false, "only implemented for Windows");
#endif
	}

	// Before CUDA writes or reads the shared presentation image.
	static void cudaWait(CUstream stream) {
		init();
		CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS params{};
		params.params.fence.value = cudaWaitValue();
		check(cuWaitExternalSemaphoresAsync(&cudaTimeline, &params, 1, stream) == CUDA_SUCCESS, "CUDA wait failed");
	}

	// After the last CUDA access to the shared presentation image in this iteration.
	static void cudaSignal(CUstream stream) {
		init();
		CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS params{};
		params.params.fence.value = cudaSignalValue();
		check(cuSignalExternalSemaphoresAsync(&cudaTimeline, &params, 1, stream) == CUDA_SUCCESS, "CUDA signal failed");
		cudaSignaled = true;
	}

	// Vulkan submit: wait for CUDA only if it signaled this iteration; always signal the blit value.
	static bool vulkanWaits() { return timeline != VK_NULL_HANDLE && cudaSignaled; }

	// End of a loop iteration. Without a Vulkan submit, wait for CUDA and advance the timeline on the host.
	static void endIteration() {
		if (timeline != VK_NULL_HANDLE && !vulkanSignaled) {
			uint64_t value = cudaSignaled ? cudaSignalValue() : cudaWaitValue();
			VkSemaphoreWaitInfo waitInfo = {};
			waitInfo.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
			waitInfo.semaphoreCount = 1;
			waitInfo.pSemaphores    = &timeline;
			waitInfo.pValues        = &value;
			check(vkWaitSemaphores(VKRenderer::device, &waitInfo, UINT64_MAX) == VK_SUCCESS, "host wait failed");

			VkSemaphoreSignalInfo signalInfo = {};
			signalInfo.sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
			signalInfo.semaphore = timeline;
			signalInfo.value     = vulkanSignalValue();
			check(vkSignalSemaphore(VKRenderer::device, &signalInfo) == VK_SUCCESS, "host signal failed");
		}
		cudaSignaled = false;
		vulkanSignaled = false;
		iteration++;
	}

	static void destroy() {
		if (cudaTimeline) { cuDestroyExternalSemaphore(cudaTimeline); cudaTimeline = nullptr; }
		if (timeline != VK_NULL_HANDLE) { vkDestroySemaphore(VKRenderer::device, timeline, nullptr); timeline = VK_NULL_HANDLE; }
	}
};
