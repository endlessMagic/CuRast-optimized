#pragma once

#include <vector>
#include <queue>
#include <string>
#include <cstdio>
#include <cstdlib>

#include "cuda.h"
#include <vulkan/vulkan.h>



using namespace std;

struct Timer{

	struct Timestamp{
		CUevent  cudaEvent  = nullptr;
		uint32_t vkQueryIdx = UINT32_MAX; // UINT32_MAX = CUDA timestamp (not Vulkan)
	};

	struct Recording{
		Timestamp start;
		Timestamp end;
		string label;
		double milliseconds;
	};

	// ---- CUDA ----
	inline static queue<Timestamp> pool;
	inline static bool enabled = false;
	inline static vector<Timestamp> timestamps;
	inline static vector<Recording> recordings;

	static void checkCuda(CUresult result){
		if(result == CUDA_SUCCESS) return;
		const char* description = nullptr;
		cuGetErrorString(result, &description);
		fprintf(stderr, "CUDA timing/launch error %d: %s\n", int(result), description ? description : "unknown");
		exit(1);
	}

	// ---- Vulkan ----
	static constexpr uint32_t VK_QUERIES_PER_FRAME = 10;
	inline static VkQueryPool  vkPool              = VK_NULL_HANDLE;
	inline static float        vkTimestampPeriod   = 1.0f; // nanoseconds per tick
	inline static vector<uint32_t>          vkNextSlot;
	inline static vector<vector<Recording>> vkPendingRecordings;

	static void init(){
		static bool initialized = false;

		if(!initialized){

			int poolSize = 1000;

			for(int i = 0; i < poolSize; i++){

				Timestamp timestamp;

				checkCuda(cuEventCreate(&timestamp.cudaEvent, CU_EVENT_DEFAULT));

				pool.push(timestamp);
			}

			initialized = true;
		}
	}

	static Timestamp recordCudaTimestamp(CUstream stream = 0){

		if(!enabled) return Timestamp();
		init();

		Timestamp timestamp;
		if(pool.empty()){
			checkCuda(cuEventCreate(&timestamp.cudaEvent, CU_EVENT_DEFAULT));
		}else{
			timestamp = pool.front();
			pool.pop();
		}

		checkCuda(cuEventRecord(timestamp.cudaEvent, stream));

		timestamps.push_back(timestamp);

		return timestamp;
	}

	static void recordDuration(string label, Timestamp start, Timestamp end){

		if(!enabled) return;
		init();

		Recording recording;
		recording.label = label;
		recording.start = start;
		recording.end = end;

		recordings.push_back(recording);
	}

	inline static vector<Timestamp> deferredTimestamps;
	inline static vector<Recording> deferredRecordings;

	// Pipelined frames: return the previous frame's recordings, waiting only for their events,
	// and keep this frame's recordings pending. A second call after the last frame flushes it.
	static vector<Recording> resolveDeferred(){
		vector<Recording> previous = std::move(deferredRecordings);
		vector<Timestamp> previousTimestamps = std::move(deferredTimestamps);
		deferredRecordings = std::move(recordings);
		deferredTimestamps = std::move(timestamps);
		recordings.clear();
		timestamps.clear();

		for(Recording& recording : previous){
			checkCuda(cuEventSynchronize(recording.end.cudaEvent));
			float duration;
			checkCuda(cuEventElapsedTime(&duration, recording.start.cudaEvent, recording.end.cudaEvent));
			recording.milliseconds = duration;
		}

		for(Timestamp timestamp : previousTimestamps){
			pool.push(timestamp);
		}

		return previous;
	}

	// Evaluate all pending timestamp queries, then clear them and put them back into the pool
	static vector<Recording> resolve(){

		if(recordings.empty()) return vector<Recording>();
		checkCuda(cuCtxSynchronize());

		for(Recording& recording : recordings){
			float duration;
			checkCuda(cuEventElapsedTime(&duration, recording.start.cudaEvent, recording.end.cudaEvent));

			recording.milliseconds = duration;
		}

		for(Timestamp timestamp : timestamps){
			pool.push(timestamp);
		}

		vector<Recording> returnvalue = recordings;

		timestamps.clear();
		recordings.clear();

		return returnvalue;
	}

	// ---- Vulkan GPU timestamps ----

	// Call once after device creation.
	static void initVulkan(VkPhysicalDevice physDev, VkDevice device, uint32_t framesInFlight){
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(physDev, &props);
		vkTimestampPeriod = props.limits.timestampPeriod;

		vkNextSlot.assign(framesInFlight, 0);
		vkPendingRecordings.resize(framesInFlight);

		VkQueryPoolCreateInfo qpci = {};
		qpci.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
		qpci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
		qpci.queryCount = VK_QUERIES_PER_FRAME * framesInFlight;
		vkCreateQueryPool(device, &qpci, nullptr, &vkPool);
	}

	// Call before device destruction.
	static void destroyVulkan(VkDevice device){
		if(vkPool != VK_NULL_HANDLE){
			vkDestroyQueryPool(device, vkPool, nullptr);
			vkPool = VK_NULL_HANDLE;
		}
	}

	// Call at the start of command buffer recording to reset this frame slot's queries.
	static void resetVulkanFrame(VkCommandBuffer cmd, int frame){
		if(!enabled || vkPool == VK_NULL_HANDLE) return;
		vkNextSlot[frame] = 0;
		vkPendingRecordings[frame].clear();
		vkCmdResetQueryPool(cmd, vkPool, (uint32_t)frame * VK_QUERIES_PER_FRAME, VK_QUERIES_PER_FRAME);
	}

	// Records a GPU timestamp into the command buffer and returns a handle to it.
	static Timestamp recordVulkanTimestamp(VkCommandBuffer cmd, int frame){
		if(!enabled || vkPool == VK_NULL_HANDLE) return Timestamp{};
		Timestamp ts;
		ts.vkQueryIdx = (uint32_t)frame * VK_QUERIES_PER_FRAME + vkNextSlot[frame]++;
		vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, vkPool, ts.vkQueryIdx);
		return ts;
	}

	static void recordVulkanDuration(string label, Timestamp start, Timestamp end, int frame){
		if(!enabled) return;
		Recording r;
		r.label = label;
		r.start = start;
		r.end   = end;
		vkPendingRecordings[frame].push_back(r);
	}

	// Call after waitForFences(currentFrame) to read back Vulkan timings for that frame slot.
	static vector<Recording> resolveVulkan(VkDevice device, int frame){
		if(!enabled || vkPool == VK_NULL_HANDLE || vkPendingRecordings[frame].empty()) return {};

		uint32_t base  = (uint32_t)frame * VK_QUERIES_PER_FRAME;
		uint32_t count = vkNextSlot[frame];
		if(count == 0) return {};

		vector<uint64_t> results(count, 0);
		vkGetQueryPoolResults(
			device, vkPool, base, count,
			count * sizeof(uint64_t), results.data(), sizeof(uint64_t),
			VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
		);

		vector<Recording> out;
		for(auto& r : vkPendingRecordings[frame]){
			uint32_t startSlot = r.start.vkQueryIdx - base;
			uint32_t endSlot   = r.end.vkQueryIdx   - base;
			if(startSlot < count && endSlot < count){
				r.milliseconds = double(results[endSlot] - results[startSlot]) * vkTimestampPeriod * 1e-6;
				out.push_back(r);
			}
		}
		vkPendingRecordings[frame].clear();
		return out;
	}

};
