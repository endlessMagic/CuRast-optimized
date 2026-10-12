#include <cstdio>
#include <format>
#include <print>
#include <filesystem>
#include <string>
#include <queue>
#include <vector>
#include <algorithm>
#include <execution>
#include <thread>
#include <charconv>
#include <limits>

#include "unsuck.hpp"

#include "cuda.h"
#include "cuda_runtime.h"
#include "CudaModularProgram.h"
#include "CudaVulkanFrameSync.h"
#include "CudaVulkanSharedMemory.h"
#include "VulkanCudaSharedMemory.h"
#include "jpeg/JPEGIndexer.h"

#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include "Runtime.h"
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include "json/json.hpp"
#include "CuRast.h"
#include "MappedFile.h"
#include "GLTFLoader.h"
#include "LargeGlbLoader.h"
#include "PlyLoader.h"
#include "BenchmarkRun.h"



using namespace std; // YOLO

CUcontext context;

mat4 flip = mat4(
	1.000,  0.000, 0.000, 0.000,
	0.000,  0.000, 1.000, 0.000,
	0.000, -1.000, 0.000, 0.000,
	0.000,  0.000, 0.000, 1.000);

void initCuda() {
	cuInit(0);
	
	CUctxCreateParams creation_params = {};
	cuDeviceGet(&CURuntime::device, 0);
	cuCtxCreate(&context, &creation_params, 0, CURuntime::device);
}

std::string benchmarkAssetPath() {
	return BenchmarkRun::fixture == "donaukanal" ? "resources/example_donaukanal_urania.glb" : "resources/hakone_1M.glb";
}

shared_ptr<largeGlb::LoadedGlb> loadFixtureAsset(const std::string& file) {
	return largeGlb::load(file, context, {.skipUVs = false, .compress = false});
}

void initScene() {
	CuRast* editor = CuRast::instance;
	Scene& scene = editor->scene;

	// position: 124.54672426747658, -42.72048538939598, -12.2730454323992 
	Runtime::controls->yaw    = -5.179;
	Runtime::controls->pitch  = 0.108;
	Runtime::controls->radius = 142.656;
	Runtime::controls->target = { -2.859, 21.085, -5.387, };

	auto loadSponza = [=](){ 
		string file = "F:/resources/meshes/sponza-png_by_Ludicon.glb";

		static auto glb = largeGlb::load(file, context, {.compress = false});
		glb->glbNode->name = "Sponza";
		glb->glbNode->transform = flip * glb->glbNode->transform;
		scene.world->children.push_back(glb->glbNode);

		Runtime::controls->yaw    = -4.731;
		Runtime::controls->pitch  = 0.009;
		Runtime::controls->radius = 336.359;
		Runtime::controls->target = { -2.986, 32.881, 119.491, };
	};

	auto loadSponzaJPEG = [=](){ 
		std::string file = "./resources/meshes/Sponza_70.glb";

		static auto glb = largeGlb::load(file, context, {
			.skipUVs = false, 
			.compress = true,
			.useJpegTextures = true,
		});
		glb->glbNode->transform = flip * glb->glbNode->transform;
		scene.world->children.push_back(glb->glbNode);

		// position: 1.3305474790692626, 0.45402304811990946, 1.1192273142715552 
		Runtime::controls->yaw    = -4.646;
		Runtime::controls->pitch  = 0.024;
		Runtime::controls->radius = 8.993;
		Runtime::controls->target = { -7.642, -0.142, 1.105, };
	};

	auto loadCubeJpeg = [=](){ 
		std::string file = "./resources/meshes/Cube_70.glb";

		static auto glb = largeGlb::load(file, context, {
			.skipUVs = false, 
			.compress = true,
			.useJpegTextures = true,
		});
		glb->glbNode->transform = flip * glb->glbNode->transform;
		editor->scene.world->children.push_back(glb->glbNode);

		// position: 25.15382712255294, -20.17937489109018, 12.02894404906026 
		Runtime::controls->yaw    = -5.466;
		Runtime::controls->pitch  = -0.531;
		Runtime::controls->radius = 34.148;
		Runtime::controls->target = { 0.252, -0.030, 0.196};
	};

	auto loadHakone = [=](){
		// string file = "./resources/meshes/donaukanal_urania_1M_jpeg80.glb";
		// string file = "F:/resources/meshes/hakone_lantern.glb";
		string file = "./resources/hakone_1M.glb";

		static auto glb = loadFixtureAsset(file);
		editor->scene.world->children.push_back(glb->glbNode);
		if(BenchmarkRun::enabled && BenchmarkRun::fixture == "hakone-instances"){
			BenchmarkRun::require(glb->glbNode->children.size() == 1, "Hakone instance fixture requires one mesh node");
			auto original = dynamic_pointer_cast<SNTriangles>(glb->glbNode->children[0]);
			BenchmarkRun::require(original && original->children.empty(), "Hakone instance fixture requires a flat mesh node");
			// Three instances (one translated, one reflected) sharing the same geometry and texture.
			const std::vector<mat4> transforms = {
				mat4(1.0f),
				glm::translate(vec3{0.5506777000427246f, 0.0f, 0.0f}),
				glm::translate(vec3{49.109172668457035f, 0.0f, 0.0f}) * glm::scale(vec3{-1.0f, 1.0f, 1.0f})
			};
			glb->glbNode->children.clear();
			for(const auto& transform : transforms){
				auto instance = make_shared<SNTriangles>("benchmark instance");
				instance->mesh = original->mesh;
				instance->texture = original->texture;
				instance->aabb = original->aabb;
				instance->transform = transform;
				glb->glbNode->children.push_back(instance);
			}
		}

		// Overview
		Runtime::controls->yaw    = -7.070;
		Runtime::controls->pitch  = -0.515;
		Runtime::controls->radius = 37.564;
		Runtime::controls->target = { 25.607, -17.328, 8.340, };
	};

	auto loadBenchmarkDonaukanal = [=](){
		static auto glb = loadFixtureAsset("./resources/example_donaukanal_urania.glb");
		editor->scene.world->children.push_back(glb->glbNode);
		// View the textured street/canal side above the coarse outer scan faces.
		Runtime::controls->yaw    = -4.062407346410207;
		Runtime::controls->pitch  = -0.9;
		Runtime::controls->radius = 482.00092684918354;
		Runtime::controls->target = { 36.32972717285156, 120.40862274169922, 29.570054054260254 };
	};

	auto loadHakoneInstances = [=](){
		// string file = "./resources/meshes/donaukanal_urania_1M_jpeg80.glb";
		// string file = "F:/resources/meshes/hakone_lantern.glb";
		// string file = "F:/resources/meshes/hakone_lantern_optimized.glb";
		// string file = "F:/resources/meshes/hakone_lantern_3.glb";
		// string file = "F:/resources/meshes/hakone_1m.glb";
		string file = "F:/resources/meshes/hakone_1m_optimized.glb";

		static auto glb = largeGlb::load(file, context, {.skipUVs = false, .compress = false});

		shared_ptr<SNTriangles> original = dynamic_pointer_cast<SNTriangles>(glb->glbNode->children[0]);

		for(int ix = 0; ix < 50; ix++)
		for(int iy = 0; iy < 60; iy++)
		{
			shared_ptr<SNTriangles> instance = make_shared<SNTriangles>("instance");
			instance->mesh = original->mesh;
			instance->texture = original->texture;
			instance->aabb = original->aabb;
			instance->transform = glm::translate(vec3{ix * 30.0f, iy * 30.0f, 0.0f});
			editor->scene.world->children.push_back(instance);
		}

		// position: -0.9794631786208647, -40.41708964196321, 21.421843904392993 
		Runtime::controls->yaw    = -7.070;
		Runtime::controls->pitch  = -0.515;
		Runtime::controls->radius = 37.564;
		Runtime::controls->target = { 25.607, -17.328, 8.340, };
	};

	auto loadSpot = [=](){ 
		std::string file = "F:/resources/meshes/spot.glb";

		static auto glb = largeGlb::load(file, context, {.skipUVs = false, .compress = false});
		editor->scene.world->children.push_back(glb->glbNode);

		// position: 1.1620903300458982, 1.4676847017158816, -0.27796041598897114 
		Runtime::controls->yaw    = -16.358;
		Runtime::controls->pitch  = -0.381;
		Runtime::controls->radius = 1.957;
		Runtime::controls->target = { -0.023, 0.022, 0.301, };
	};

	auto loadZorah = [=](){ 
		string file = "F:/resources/meshes/zorah_main_public.gltf/zorah_main_public.gltf";
		// string file = "F:/resources/meshes/zorah_main_public.gltf_optimized/zorah_main_public.gltf";
		
		// Mesh has no textures/uvs/normals
		CuRastSettings::displayAttribute = DisplayAttribute::NONE;

		static auto glb = largeGlb::load(file, context, {.skipUVs = true, .compress = true});
		glb->glbNode->transform = flip;
		editor->scene.world->children.push_back(glb->glbNode);

		// Let's remove some less appealing billboards
		vector<shared_ptr<SceneNode>> filtered;
		for(shared_ptr<SceneNode> node : glb->glbNode->children){
			if(node->name == "FogCard") continue;
			if(node->name == "Plane") continue;
			
			filtered.push_back(node);
		}
		glb->glbNode->children = filtered;

		// Overview
		Runtime::controls->yaw    = -16.537;
		Runtime::controls->pitch  = -0.472;
		Runtime::controls->radius = 97.539;
		Runtime::controls->target = { 17.436, -6.343, 3.689, };
		
		// Closeup
		Runtime::controls->yaw    = -17.344;
		Runtime::controls->pitch  = 0.073;
		Runtime::controls->radius = 10.893;
		Runtime::controls->target = { 44.294, 1.156, 6.458, };

	};

	auto createCube = [&](){
		static shared_ptr<SNTriangles> node = make_shared<SNTriangles>("node");
		node->texture = new Texture();
		node->mesh = new Mesh();

		{ // Create Default Texture
			int64_t textureWidth = 128;
			int64_t textureHeight = 128;
			vector<uint8_t> textureData = vector<uint8_t>(2 * 4 * textureWidth * textureHeight, 255);
			
			node->texture->width = textureWidth;
			node->texture->height = textureHeight;
			node->texture->data = (uint32_t*)MemoryManager::alloc(byteSizeOf(textureData), "default texture");

			cuMemcpyHtoDAsync(CUdeviceptr(node->texture->data), textureData.data(), byteSizeOf(textureData), 0);
		}

		node->mesh->isLoaded = true;
		node->mesh->name = "default mesh";
		node->mesh->numTriangles = 0;

		vector<vec3> positions = {
			vec3{0.0f, 0.0f, 0.0f},
			vec3{1.0f, 0.0f, 0.0f},
			vec3{1.0f, 1.0f, 0.0f},
			vec3{0.0f, 1.0f, 0.0f},
		};
		vector<vec2> uvs = {
			vec2{0.0f, 0.0f},
			vec2{1.0f, 0.0f},
			vec2{1.0f, 1.0f},
			vec2{0.0f, 1.0f},
		};
		vector<uint32_t> indices = {0, 1, 2, 0, 2, 3};

		int numVertices = positions.size();
		int numTriangles = indices.size() / 3;
		node->mesh->cptr_position = MemoryManager::alloc(sizeof(vec3) * numVertices, "position");
		node->mesh->cptr_uv       = MemoryManager::alloc(sizeof(vec2) * numVertices, "uv");
		node->mesh->cptr_indices  = MemoryManager::alloc(sizeof(uint32_t) * 3 * numTriangles, "indices");
		node->aabb.extend(vec3{-1.0f, -1.0f, -1.0f});
		node->aabb.extend(vec3{1.0f, 1.0f, 1.0f});

		cuMemcpyHtoD(node->mesh->cptr_position, positions.data(), byteSizeOf(positions));
		cuMemcpyHtoD(node->mesh->cptr_uv, uvs.data(), byteSizeOf(uvs));
		cuMemcpyHtoD(node->mesh->cptr_indices, indices.data(), byteSizeOf(indices));

		node->mesh->numTriangles = indices.size() / 3;
		node->mesh->numVertices = positions.size();

		scene.world->children.push_back(node);

		CuRastSettings::displayAttribute = DisplayAttribute::TRIANGLE_ID;

		// position: 0.8369693760957783, 0.05588397571280396, 0.02743282811653472 
		Runtime::controls->yaw    = -17.426;
		Runtime::controls->pitch  = -0.272;
		Runtime::controls->radius = 0.818;
		Runtime::controls->target = { 0.028, 0.172, -0.005, };
	};

	auto loadVenice = [=](){ 
		string file = "F:/resources/meshes/iconem/VeniceGeneral-Airborne-flyover-400M-12x16k-local-binply/venice.gltf";
		// string file = "F:/resources/meshes/iconem/VeniceGeneral-Airborne-flyover-400M-12x16k-local-binply/venice_optimized.gltf";
		
		CuRastSettings::displayAttribute = DisplayAttribute::TEXTURE;

		static auto glb = largeGlb::load(file, context, {
			.skipNormals = true, // We need that previous VRAM
			.skipVertexColors = true,
			.compress = true,
			.useJpegTextures = false,
			.imageDivisionFactor = 2
		});
		editor->scene.world->children.push_back(glb->glbNode);
		
		// Distant
		Runtime::controls->yaw    = -18.968;
		Runtime::controls->pitch  = -0.769;
		Runtime::controls->radius = 4180.978;
		Runtime::controls->target = { 352.960, -1134.931, -462.529, };
	};

	// createCube();
	// loadZorah();
	// loadGraffiti();
	// loadHakone();
	// loadHakoneInstances();
	// loadXyzDragon();
	// loadSpot();
	// loadWietrznia();
	// loadSponza();
	// loadSponzaJPEG();
	// loadCubeJpeg();
	// loadPolygraphenewerkLeibzigInstances();
	// loadVenice();

	// Benchmark fixtures (--benchmark-fixture hakone|hakone-instances|donaukanal).
	if(BenchmarkRun::enabled){
		if(BenchmarkRun::fixture == "donaukanal") loadBenchmarkDonaukanal();
		else loadHakone();
	}

}

void update(){

	if(Benchmarking::request_scenario){

		auto scenario = Benchmarking::request_scenario;

		string path = stringReplace(scenario->path, "DATASETPATH", Benchmarking::datasetPath);

		CuRastSettings::displayAttribute = scenario->attribute;

		static auto glb = largeGlb::load(path, context, {
			.skipUVs = scenario->skipUVs,
			.skipNormals = scenario->skipNormals,
			.compress = scenario->compress,
			.useJpegTextures = scenario->useJpegTextures,
			.imageDivisionFactor = scenario->imageDivisionFactor,
		});
		glb->glbNode->name = scenario->label;
		glb->glbNode->transform = scenario->transform * glb->glbNode->transform;

		vector<shared_ptr<SceneNode>> filtered;
		for(shared_ptr<SceneNode> node : glb->glbNode->children){

			bool accept = scenario->filter(node);
			
			if(accept){
				filtered.push_back(node);
			}
		}
		glb->glbNode->children = filtered;

		shared_ptr<SceneNode> original = glb->glbNode;

		function<shared_ptr<SceneNode>(shared_ptr<SceneNode>)> deepClone =
		[&deepClone](shared_ptr<SceneNode> node) -> shared_ptr<SceneNode> {

			shared_ptr<SceneNode> clone;

			shared_ptr<SNTriangles> tris = dynamic_pointer_cast<SNTriangles>(node);
			if(tris){
				auto triClone      = make_shared<SNTriangles>(tris->name);
				triClone->mesh     = tris->mesh;
				triClone->texture  = tris->texture;
				triClone->aabb     = tris->aabb;
				clone = triClone;
			}else{
				clone = make_shared<SceneNode>(node->name);
				clone->aabb = node->aabb;
			}

			clone->transform = node->transform;
			clone->visible   = node->visible;

			for(auto& child : node->children){
				clone->children.push_back(deepClone(child));
			}

			return clone;
		};

		for(int ix = 0; ix < scenario->instances_count.x; ix++)
		for(int iy = 0; iy < scenario->instances_count.y; iy++)
		{
			shared_ptr<SceneNode> clone = deepClone(original);
			clone->transform = glm::translate(vec3{ix * scenario->instances_spacing.x, iy * scenario->instances_spacing.y, 0.0f}) * clone->transform;
			CuRast::instance->scene.world->children.push_back(clone);
		}

		Runtime::controls->yaw    = scenario->view_overview.yaw;
		Runtime::controls->pitch  = scenario->view_overview.pitch;
		Runtime::controls->radius = scenario->view_overview.radius;
		Runtime::controls->target = scenario->view_overview.target;

		Benchmarking::active_scenario = Benchmarking::request_scenario;
		Benchmarking::request_scenario = nullptr;
	}

}

void recordBenchmarkRuntime() {
	CUdevice device;
	Timer::checkCuda(cuCtxGetDevice(&device));
	char name[256] = {};
	Timer::checkCuda(cuDeviceGetName(name, sizeof(name), device));
	CUuuid uuid;
	Timer::checkCuda(cuDeviceGetUuid(&uuid, device));
	std::string uuidText;
	for(unsigned char byte : uuid.bytes) uuidText += std::format("{:02x}", byte);
	int driver = 0, major = 0, minor = 0, rtcMajor = 0, rtcMinor = 0;
	Timer::checkCuda(cuDriverGetVersion(&driver));
	Timer::checkCuda(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
	Timer::checkCuda(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
	BenchmarkRun::require(nvrtcVersion(&rtcMajor, &rtcMinor) == NVRTC_SUCCESS, "cannot query NVRTC version");
	unsigned int jitMajor = 0, jitMinor = 0;
	BenchmarkRun::require(nvJitLinkVersion(&jitMajor, &jitMinor) == NVJITLINK_SUCCESS, "cannot query nvJitLink version");
	BenchmarkRun::runtime["gpu_name"] = name;
	BenchmarkRun::runtime["gpu_uuid"] = uuidText;
	BenchmarkRun::runtime["driver_version"] = driver;
	BenchmarkRun::runtime["compute_capability"] = {major, minor};
	BenchmarkRun::runtime["nvrtc_version"] = {rtcMajor, rtcMinor};
	BenchmarkRun::runtime["nvjitlink_version"] = {jitMajor, jitMinor};
	BenchmarkRun::runtime["cuda_devrtlib"] = CUDA_DEVRTLIB;
	const char* cache = std::getenv("CUDA_CACHE_DISABLE");
	BenchmarkRun::runtime["cuda_cache_disable"] = cache ? nlohmann::json(cache) : nlohmann::json(nullptr);
}

struct CommandLineOptions {
	bool headless = false;
	string output;
	int nframes = 0;
	bool benchmarkOptions = false;
	bool benchmarkFixtureSet = false;
	bool datasetOverride = false;
};

void printUsage(char** argv) {
	println("Usage: {} [-b datasetPath] [--headless] [--output path.png] [--nframes N] [--benchmark-output path.jsonl --benchmark-fixture hakone|hakone-instances|donaukanal --benchmark-warmup N --benchmark-frames N --benchmark-width N --benchmark-height N]", argv[0]);
}

CommandLineOptions parseCommandLine(int argc, char** argv) {
	CommandLineOptions options;

	for(int i = 1; i < argc; i++){
		string arg = argv[i];
		if(arg == "-b"){
			if(i + 1 >= argc){
				printUsage(argv);
				exit(1);
			}
			Benchmarking::datasetPath = argv[++i];
			options.datasetOverride = true;
		}else if(arg == "--headless"){
			options.headless = true;
		}else if(arg == "--output"){
			if(i + 1 >= argc){
				printUsage(argv);
				exit(1);
			}
			options.output = argv[++i];
		}else if(arg == "--benchmark-output"){
			BenchmarkRun::require(i + 1 < argc, "--benchmark-output requires a path");
			BenchmarkRun::require(BenchmarkRun::outputPath.empty(), "duplicate --benchmark-output");
			BenchmarkRun::outputPath = argv[++i];
			BenchmarkRun::require(!BenchmarkRun::outputPath.empty(), "empty --benchmark-output");
		}else if(arg == "--benchmark-fixture"){
			BenchmarkRun::require(i + 1 < argc, "--benchmark-fixture requires a name");
			BenchmarkRun::require(!options.benchmarkFixtureSet, "duplicate --benchmark-fixture");
			BenchmarkRun::fixture = argv[++i];
			BenchmarkRun::require(BenchmarkRun::fixture == "hakone" || BenchmarkRun::fixture == "hakone-instances" ||
				BenchmarkRun::fixture == "donaukanal", "unknown benchmark fixture: " + BenchmarkRun::fixture);
			options.benchmarkFixtureSet = true;
			options.benchmarkOptions = true;
		}else if(arg == "--benchmark-warmup" || arg == "--benchmark-frames" || arg == "--benchmark-width" || arg == "--benchmark-height"){
			BenchmarkRun::require(i + 1 < argc, arg + " requires a positive integer");
			string value = argv[++i];
			int count = 0;
			auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), count);
			BenchmarkRun::require(ec == std::errc() && ptr == value.data() + value.size() && count > 0, arg + " requires a positive integer");
			if(arg == "--benchmark-warmup") BenchmarkRun::warmupFrames = count;
			if(arg == "--benchmark-frames") BenchmarkRun::measuredFrames = count;
			if(arg == "--benchmark-width") BenchmarkRun::width = count;
			if(arg == "--benchmark-height") BenchmarkRun::height = count;
			options.benchmarkOptions = true;
		}else if(arg == "--nframes"){
			if(i + 1 >= argc){
				printUsage(argv);
				exit(1);
			}

			string value = argv[++i];
			int nframes = 0;
			auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), nframes);
			if(ec != std::errc() || ptr != value.data() + value.size() || nframes <= 0){
				println("--nframes requires a positive integer");
				printUsage(argv);
				exit(1);
			}
			options.nframes = nframes;
		}else{
			println("Unknown argument: {}", arg);
			printUsage(argv);
			exit(1);
		}
	}

	BenchmarkRun::enabled = !BenchmarkRun::outputPath.empty();
	BenchmarkRun::require(!options.benchmarkOptions || BenchmarkRun::enabled, "benchmark controls require --benchmark-output");
	if(BenchmarkRun::enabled){
		BenchmarkRun::require(options.headless, "--benchmark-output requires --headless");
		BenchmarkRun::require(options.nframes == 0, "--nframes conflicts with benchmark frame counts");
		BenchmarkRun::require(!options.datasetOverride, "benchmark fixtures do not use -b dataset selection");
		BenchmarkRun::require(int64_t(BenchmarkRun::warmupFrames) + BenchmarkRun::measuredFrames <= std::numeric_limits<int>::max(), "frame count overflow");
		BenchmarkRun::require(BenchmarkRun::width <= 4096 && BenchmarkRun::height <= 4096, "benchmark dimensions exceed 4096");
		BenchmarkRun::require(options.output != BenchmarkRun::outputPath, "screenshot and raw output paths must differ");
	}
	return options;
}

int main(int argc, char** argv){
	BenchmarkRun::processStart = std::chrono::steady_clock::now();

	// With the CUDA cache enabled, nvJitLink can crash intermittently while the kernels are linked at start-up.
	// Disable the cache unless the user set CUDA_CACHE_DISABLE explicitly. This must run before CUDA is used;
	// _putenv_s also updates the process environment that the CUDA libraries read.
	if(std::getenv("CUDA_CACHE_DISABLE") == nullptr){
		_putenv_s("CUDA_CACHE_DISABLE", "1");
	}
	try {

	Benchmarking::datasetPath = "./";

	CommandLineOptions options = parseCommandLine(argc, argv);
	if(BenchmarkRun::enabled) CudaModularProgram::watchSourceFiles = false;
	std::locale::global(getSaneLocale());

	initCuda();
	VKRenderer::headless = options.headless;
	VKRenderer::init();
	CuRast::setup();

	VKRenderer::onFileDrop([](vector<string> files){


		if(BenchmarkRun::enabled || files.size() != 1) return;

		CuRast* editor = CuRast::instance;
		Scene& scene = editor->scene;

		string file = files[0];

		static vector<shared_ptr<largeGlb::LoadedGlb>> loadedGlbs;

		if(iEndsWith(file, ".gltf") || iEndsWith(file, ".glb")){
			Scene& scene = editor->scene;

			auto glb = largeGlb::load(file, context, {.skipUVs = false, .compress = false});
			scene.world->children.push_back(glb->glbNode);
			loadedGlbs.push_back(glb);

			scene.updateTransformations();

			Box3 aabb = glb->glbNode->aabb;
			vec3 extent = aabb.max - aabb.min;
			vec3 center = (aabb.min + aabb.max) * 0.5f;

			Runtime::controls->yaw    = -7.204;
			Runtime::controls->pitch  = -0.579;
			Runtime::controls->radius = length(extent);
			Runtime::controls->target = { center.x, center.y, center.z};
		}

	});

	initScene();
	if(BenchmarkRun::enabled){
		recordBenchmarkRuntime();
		BenchmarkRun::begin();
	}

	int nframes = BenchmarkRun::enabled ? BenchmarkRun::warmupFrames + BenchmarkRun::measuredFrames : options.nframes;
	if(nframes == 0 && !options.output.empty()){
		nframes = 1;
	}

	VKRenderer::loop(
		[&]() {
			if(!BenchmarkRun::enabled) update();
			CuRast::instance->update();
			
			DeviceState* state = CuRast::instance->deviceState;
			double stage1_millies = double(state->nanotime_stage_1 - state->nanotime_start) / 1'000'000.0;
			double stage2_millies = double(state->nanotime_stage_2 - state->nanotime_stage_1) / 1'000'000.0;
			double stage3_millies = double(state->nanotime_stage_3 - state->nanotime_stage_2) / 1'000'000.0;
			Runtime::debugValues["stage 1"] = format("{:.3f}", stage1_millies);
			Runtime::debugValues["stage 2"] = format("{:.3f}", stage2_millies);
			Runtime::debugValues["stage 3"] = format("{:.3f}", stage3_millies);
		},
		[&]() {
			int64_t frameNumber = VKRenderer::frameCount + 1;
			if(!options.output.empty() && frameNumber == nframes){
				CuRastSettings::requestScreenshot = make_shared<string>(options.output);
			}
			CuRast::instance->render();
		},
		[&]() {
			CuRast::instance->postFrame();
			int64_t frameNumber = VKRenderer::frameCount + 1;
			if(nframes > 0 && frameNumber >= nframes){
				glfwSetWindowShouldClose(VKRenderer::window, GLFW_TRUE);
			}
		}
	);

	if(BenchmarkRun::enabled) BenchmarkRun::complete();
	VKRenderer::destroy();
	return 0;
	} catch(const std::exception& error) {
		fprintf(stderr, "%s\n", error.what());
		return 1;
	}
}
