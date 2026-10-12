#pragma once

#include <cmath>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "json/json.hpp"
#include "Timer.h"

// Raw completed CUDA batches only; GUI histories and delayed Vulkan queries are separate.
struct BenchmarkRun {
	inline static bool enabled = false;
	inline static int warmupFrames = 60;
	inline static int measuredFrames = 200;
	inline static int width = 1920;
	inline static int height = 1080;
	inline static std::string fixture = "hakone";
	inline static std::string outputPath;
	inline static nlohmann::json runtime = nlohmann::json::object();
	inline static std::chrono::steady_clock::time_point processStart;
	inline static nlohmann::json config;
	inline static std::ofstream output;
	inline static bool configWritten = false;
	inline static int sampledFrames = 0;
	inline static int64_t nextFrame = 0;
	inline static std::vector<std::string> timingLabels;

	static void require(bool condition, const std::string& message) {
		if (!condition) throw std::runtime_error("benchmark: " + message);
	}

	static void write(const nlohmann::json& record) {
		output << record.dump() << '\n';
		output.flush();
		require(bool(output), "cannot write raw output");
	}

	static void begin() {
		require(!std::filesystem::exists(outputPath), "output already exists: " + outputPath);
		auto parent = std::filesystem::path(outputPath).parent_path();
		if (!parent.empty()) std::filesystem::create_directories(parent);
		output.open(outputPath, std::ios::out | std::ios::binary);
		require(bool(output), "cannot open output: " + outputPath);
		config = nlohmann::json::object();
		config["schema_version"] = 1;
		config["type"] = "config";
		config["fixture"] = fixture;
		config["warmup_frames"] = warmupFrames;
		config["measured_frames"] = measuredFrames;
		config["width"] = width;
		config["height"] = height;
		config["frame_timestamp_boundary"] = "CPU completion markers captured at append entry immediately after Timer::resolve, before workload validation and JSONL export; completed_unix_ns is system_clock nanoseconds since Unix epoch (wall clock may adjust), completed_elapsed_seconds is steady_clock seconds since main entry; both lie outside CUDA intervals";
	}

	static void append(int64_t frameId, const std::vector<Timer::Recording>& recordings) {
		if (!enabled) return;
		auto resolvedAt = std::chrono::steady_clock::now();
		auto completedUnixNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::system_clock::now().time_since_epoch()).count();
		double completedElapsedSeconds = std::chrono::duration<double>(resolvedAt - processStart).count();
		require(frameId == nextFrame++, "rendered frame sequence changed");
		if (!configWritten) {
			config["startup_seconds"] = std::chrono::duration<double>(resolvedAt - processStart).count();
			config["startup_boundary"] = "CPU monotonic time from main entry through first resolved CUDA frame; includes command parsing, CUDA/Vulkan initialization, scene loading, JIT and first warmup render; excludes subsequent warmup and measured frames";
			config["runtime"] = runtime;
			write(config);
			configWritten = true;
		}
		if (frameId < warmupFrames) return;
		require(sampledFrames < measuredFrames, "too many sampled frames");
		require(!recordings.empty(), "missing CUDA recordings");
		std::map<std::string, int> ordinals;
		std::vector<std::string> labels;
		nlohmann::json timings = nlohmann::json::array();
		for (const auto& recording : recordings) {
			require(!recording.label.empty() && std::isfinite(recording.milliseconds) && recording.milliseconds >= 0,
				"invalid CUDA recording");
			// An optional final screenshot is exported but lies outside the render interval.
			if (recording.label != "kernel_resolve_colorbuffer_to_screenshot") labels.push_back(recording.label);
			timings.push_back({{"label", recording.label}, {"launch_ordinal", ++ordinals[recording.label]},
				{"milliseconds", recording.milliseconds}});
		}
		require(ordinals["<CUDA render interval>"] == 1, "missing or duplicate CUDA render interval");
		if (sampledFrames == 0) timingLabels = labels;
		else require(timingLabels == labels, "CUDA launch sequence changed during collection");
		write({{"schema_version", 1}, {"type", "frame"}, {"frame_id", frameId},
			{"sample_index", sampledFrames}, {"completed_unix_ns", completedUnixNs},
			{"completed_elapsed_seconds", completedElapsedSeconds}, {"timings", timings}});
		sampledFrames++;
	}

	static void complete() {
		require(configWritten && sampledFrames == measuredFrames && nextFrame == int64_t(warmupFrames) + measuredFrames,
			"incomplete frame collection");
		nlohmann::json complete{{"schema_version", 1}, {"type", "complete"}, {"sampled_frames", sampledFrames}};
		write(complete);
		output.close();
		require(!output.fail(), "cannot close raw output");
	}
};
