// Heavy small-triangle deferral.
// Stage 1 skips small triangles whose bounding box covers more than DEFER_THRESHOLD fragments and
// lists them; Stage 2 warps rasterize the list after the medium-triangle queue is exhausted, in
// groups of 8 lanes with rows split 8 ways. Every sample sees the same s/t recurrences, inverse
// depth, pack_pixel and atomicMin as the default Stage 1 loop, so output is unchanged.

namespace heavydefer {

constexpr int DEFER_THRESHOLD = 48;

struct SmallSetup {
	int size_x, size_y;
	float s_row_start, t_row_start;
	int row_pixel_id;
	float ds_dx, dt_dx, ds_dy, dt_dy;
	float inv_z_a, inv_z_b, inv_z_c;
	uint64_t triangleId;
};

// Rows rowBegin, rowBegin + rowStride, ... of a small triangle. Row starts are advanced with the
// same per-row additions as the default loop.
__device__ __forceinline__ void rasterRows(const RasterArgs& args, const SmallSetup& q, int rowBegin, int rowStride){
	uint32_t numPixels = args.target.width * args.target.height;
	float s_row_start = q.s_row_start;
	float t_row_start = q.t_row_start;
	int row_pixel_id = q.row_pixel_id;

	for (int r = 0; r < rowBegin; r++){
		s_row_start += q.ds_dy;
		t_row_start += q.dt_dy;
		row_pixel_id += args.target.width;
	}

	for (int fragY = rowBegin; fragY < q.size_y; fragY += rowStride){
		float s = s_row_start;
		float t = t_row_start;
		int pixelID = row_pixel_id;

		for (int fragX = 0; fragX < q.size_x; fragX++){
			float v = 1.0f - (s + t);
			if (s >= 0.0f && t >= 0.0f && v >= 0.0f){
				if (pixelID < numPixels){
					float inv_depth = v * q.inv_z_a + s * q.inv_z_b + t * q.inv_z_c;
					float depth = __fdividef(1.0f, inv_depth);
					uint64_t pixel = pack_pixel(depth, q.triangleId);
					atomicMin(&args.target.framebuffer[pixelID], pixel);
				}
			}
			s += q.ds_dx;
			t += q.dt_dx;
			pixelID++;
		}

		for (int r = 0; r < rowStride; r++){
			s_row_start += q.ds_dy;
			t_row_start += q.dt_dy;
			row_pixel_id += args.target.width;
		}
	}
}

// Copy of rasterize(). With rasterNow, rasterizes small triangles immediately unless they exceed the
// deferral threshold (returns true). Without rasterNow, only fills q for a known small triangle.
__device__ __forceinline__ bool rasterize(
	const RasterArgs args, const CMesh& sh_mesh, uint32_t meshNumTriangles, const mat4& worldView,
	int triangleIndex, int instanceIndex, vec4 a_object, vec4 b_object, vec4 c_object,
	bool rasterNow, SmallSetup& q
){
	if(triangleIndex >= meshNumTriangles) return false;

	float f = args.target.proj[1][1];
	float aspect = float(args.target.width) / float(args.target.height);

	vec3 a_view = worldView * a_object;
	vec3 b_view = worldView * b_object;
	vec3 c_view = worldView * c_object;
	vec3 a_ndc = viewToNDC(a_view, f, aspect);
	vec3 b_ndc = viewToNDC(b_view, f, aspect);
	vec3 c_ndc = viewToNDC(c_view, f, aspect);

	bool isNontrivial = (a_ndc.z <= 0.0f || b_ndc.z <= 0.0f || c_ndc.z <= 0.0f);

	if(!isNontrivial){
		if(a_ndc.x > +1.0f && b_ndc.x > +1.0f && c_ndc.x > +1.0f) return false;
		if(a_ndc.x < -1.0f && b_ndc.x < -1.0f && c_ndc.x < -1.0f) return false;
		if(a_ndc.y > +1.0f && b_ndc.y > +1.0f && c_ndc.y > +1.0f) return false;
		if(a_ndc.y < -1.0f && b_ndc.y < -1.0f && c_ndc.y < -1.0f) return false;
	}
	if(a_ndc.z <= 0.0f && b_ndc.z <= 0.0f && c_ndc.z <= 0.0f) return false;

	vec2 a_screen = ndcToScreen(a_ndc, args.target.width, args.target.height);
	vec2 b_screen = ndcToScreen(b_ndc, args.target.width, args.target.height);
	vec2 c_screen = ndcToScreen(c_ndc, args.target.width, args.target.height);

	a_screen -= 0.5f;
	b_screen -= 0.5f;
	c_screen -= 0.5f;
	constexpr float SAMPLE_OFFSET = 0.0f;

	float min_x = min(a_screen.x, min(b_screen.x, c_screen.x));
	float max_x = max(a_screen.x, max(b_screen.x, c_screen.x));
	float min_y = min(a_screen.y, min(b_screen.y, c_screen.y));
	float max_y = max(a_screen.y, max(b_screen.y, c_screen.y));

	min_x = max(min_x, 0.0f);
	max_x = min(max_x, float(args.target.width - 1));
	min_y = max(min_y, 0.0f);
	max_y = min(max_y, float(args.target.height - 1));

	if(!isNontrivial){
		float sample_x = floorf(min_x);
		float sample_y = floorf(min_y);
		if(min_x > sample_x + SAMPLE_OFFSET && max_x < sample_x + 1.0f + SAMPLE_OFFSET) return false;
		if(min_y > sample_y + SAMPLE_OFFSET && max_y < sample_y + 1.0f + SAMPLE_OFFSET) return false;
	}

	int size_x = ceil(max_x) - ceil(min_x);
	int size_y = ceil(max_y) - ceil(min_y);

	int numFragments = size_x * size_y;
	if(numFragments > THRESHOLD_SMALL) isNontrivial = true;

	{
		vec3 ab = b_view - a_view;
		vec3 ac = c_view - a_view;
		vec3 N = cross(ab, ac);
		float res = dot(a_view, N);
		vec3 c0 = worldView[0];
		vec3 c1 = worldView[1];
		vec3 c2 = worldView[2];
		float det = dot(cross(c0, c1), c2);
		bool flipTriangles = det < 0.0f;
		if (flipTriangles) res = -res;
		if (res >= 0.0f) return false;
	}

	if(isNontrivial){
		uint64_t index = atomicAdd(args.nontrivialTrianglesCounter, 1);
		uint64_t packed =
			(uint64_t(sh_mesh.instances.offset + instanceIndex) << 32llu) |
			(uint64_t(triangleIndex));
		args.nontrivialTrianglesList[index] = packed;
		return false;
	}

	if (rasterNow && numFragments > DEFER_THRESHOLD) return true;

	vec2 v_ab = b_screen - a_screen;
	vec2 v_ac = c_screen - a_screen;
	float factor = __fdividef(1.0f, cross(v_ab, v_ac));

	q.inv_z_a = __fdividef(1.0f, a_ndc.z);
	q.inv_z_b = __fdividef(1.0f, b_ndc.z);
	q.inv_z_c = __fdividef(1.0f, c_ndc.z);

	q.ds_dx =  v_ac.y * factor;
	q.ds_dy = -v_ac.x * factor;
	q.dt_dx = -v_ab.y * factor;
	q.dt_dy =  v_ab.x * factor;

	float start_x = ceil(min_x);
	float start_y = ceil(min_y);
	float sample0_x = start_x - a_screen.x + SAMPLE_OFFSET;
	float sample0_y = start_y - a_screen.y + SAMPLE_OFFSET;

	q.s_row_start = (sample0_x * v_ac.y - sample0_y * v_ac.x) * factor;
	q.t_row_start = (v_ab.x * sample0_y - v_ab.y * sample0_x) * factor;
	q.row_pixel_id = toFramebufferIndex((int)start_x, (int)start_y, args.target.width);
	q.size_x = size_x;
	q.size_y = size_y;
	q.triangleId = sh_mesh.cummulativeTriangleCount + triangleIndex + instanceIndex * meshNumTriangles;

	if (rasterNow) rasterRows(args, q, 0, 1);
	return false;
}

// Deferred key: mesh (20 bits) | instance within mesh (12 bits) | triangle (32 bits).
__device__ __forceinline__ uint64_t packDeferred(int meshIndex, int instanceIndex, int triangleIndex){
	return (uint64_t(meshIndex) << 44) | (uint64_t(instanceIndex) << 32) | uint64_t(uint32_t(triangleIndex));
}

// Stage 2 consumer: groups of 8 lanes claim deferred triangles and split their rows 8 ways.
// The deferred list is complete when Stage 2 starts.
__device__ __forceinline__ void rasterizeDeferred(const RasterArgs& args){
	int lane = threadIdx.x % 32;
	int laneInGroup = lane % 8;
	unsigned groupMask = 0xffu << (lane & ~7);
	uint32_t count = *args.deferredCounter;
	while (true){
		uint32_t d = 0;
		if (laneInGroup == 0) d = atomicAdd(args.deferredProcessed, 1);
		d = __shfl_sync(groupMask, d, 0, 8);
		if (d >= count) break;

		uint64_t packed = args.deferredList[d];
		int meshIndex = int(packed >> 44);
		int instanceIndex = int((packed >> 32) & 0xfff);
		int triangleIndex = int(uint32_t(packed));
		CMesh mesh = args.meshes[meshIndex];
		uint32_t meshNumTriangles = args.meshNumTriangles[meshIndex];
		mat4 worldView = args.transforms[mesh.instances.offset + instanceIndex];
		vec4 a_object = getVertex<IndexFetch::INDEXBUFFER, Compression::UNCOMPRESSED>(mesh, 3 * triangleIndex + 0);
		vec4 b_object = getVertex<IndexFetch::INDEXBUFFER, Compression::UNCOMPRESSED>(mesh, 3 * triangleIndex + 1);
		vec4 c_object = getVertex<IndexFetch::INDEXBUFFER, Compression::UNCOMPRESSED>(mesh, 3 * triangleIndex + 2);
		SmallSetup q;
		rasterize(args, mesh, meshNumTriangles, worldView, triangleIndex, instanceIndex, a_object, b_object, c_object, false, q);
		rasterRows(args, q, laneInGroup, 8);
	}
	__syncwarp();
}

// Copy of stage1_drawSmallTriangles (ordinary indices, instancing, 128-triangle batches) with deferral.
__device__ void stage1(RasterArgs& args){
	constexpr uint32_t TRIANGLES_PER_BATCH = 128;
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	if(grid.thread_rank() == 0){
		*args.numProcessedBatches = 0;
		*args.numProcessedBatches_nontrivial = 0;
		*args.hugeTrianglesCounter = 0;
		*args.nontrivialTrianglesCounter = 0;
		*args.numProcessedHugeTriangles = 0;
		*args.deferredCounter = 0;
		*args.deferredProcessed = 0;
	}

	__shared__ int sh_blockBatchIndex;
	__shared__ int sh_blockLocalBatchIndex;
	__shared__ int sh_meshIndex;
	__shared__ uint32_t sh_meshNumTriangles;
	__shared__ CMesh sh_mesh;

	if (block.thread_rank() == 0){
		sh_blockBatchIndex = 0;
		sh_blockLocalBatchIndex = 0;
		sh_meshIndex = 0;
		sh_meshNumTriangles = args.meshNumTriangles[0];
		sh_mesh = args.meshes[0];
	}

	grid.sync();

	while (true){
		block.sync();
		if (block.thread_rank() == 0){
			uint32_t next = atomicAdd(args.numProcessedBatches, 1);
			uint32_t diff = next - sh_blockBatchIndex;
			sh_blockBatchIndex = next;
			sh_blockLocalBatchIndex += diff;

			int numBatchesInMesh = (sh_meshNumTriangles + TRIANGLES_PER_BATCH - 1) / TRIANGLES_PER_BATCH;
			while (sh_blockLocalBatchIndex >= numBatchesInMesh){
				sh_meshIndex++;
				if (sh_meshIndex >= args.numMeshes) break;
				sh_blockLocalBatchIndex -= numBatchesInMesh;
				sh_meshNumTriangles = args.meshNumTriangles[sh_meshIndex];
				sh_mesh = args.meshes[sh_meshIndex];
				numBatchesInMesh = (sh_meshNumTriangles + TRIANGLES_PER_BATCH - 1) / TRIANGLES_PER_BATCH;
			}
		}
		block.sync();

		if (sh_meshIndex >= args.numMeshes) return;
		CMesh mesh = sh_mesh;
		int meshIndex = sh_meshIndex;
		uint32_t meshNumTriangles = sh_meshNumTriangles;

		int triangleIndex = sh_blockLocalBatchIndex * TRIANGLES_PER_BATCH + block.thread_rank();
		if (triangleIndex >= meshNumTriangles) continue;

		vec4 a_object = getVertex<IndexFetch::INDEXBUFFER, Compression::UNCOMPRESSED>(mesh, 3 * triangleIndex + 0);
		vec4 b_object = getVertex<IndexFetch::INDEXBUFFER, Compression::UNCOMPRESSED>(mesh, 3 * triangleIndex + 1);
		vec4 c_object = getVertex<IndexFetch::INDEXBUFFER, Compression::UNCOMPRESSED>(mesh, 3 * triangleIndex + 2);

		for(int instanceIndex = 0; instanceIndex < mesh.instances.count; instanceIndex++){
			mat4 worldView = args.transforms[mesh.instances.offset + instanceIndex];
			SmallSetup q;
			bool deferred = rasterize(args, mesh, meshNumTriangles, worldView,
				triangleIndex, instanceIndex, a_object, b_object, c_object, true, q);
			if (deferred){
				uint32_t index = atomicAdd(args.deferredCounter, 1);
				args.deferredList[index] = packDeferred(meshIndex, instanceIndex, triangleIndex);
			}
		}
	}
}

} // namespace heavydefer
