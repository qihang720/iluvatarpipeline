#include "bytetrack_cuda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>

#ifndef BYTETRACK_CUDA_MAX_THREADS_PER_BLOCK
#define BYTETRACK_CUDA_MAX_THREADS_PER_BLOCK 256
#endif

// GPU-side track states – mirror of CPU TrackState enum.
#define TRACK_NEW     0
#define TRACK_TRACKED 1
#define TRACK_LOST    2
#define TRACK_REMOVED 3

// ============================================================
// Device utilities
// ============================================================
namespace {

__device__ __forceinline__ float clampf(float v, float lo, float hi) {
    return fmaxf(lo, fminf(v, hi));
}

// IoU of two tlbr boxes.
__device__ __forceinline__ float iou_tlbr(float a0, float a1, float a2, float a3,
                                           float b0, float b1, float b2, float b3) {
    const float iw = fmaxf(0.f, fminf(a2, b2) - fmaxf(a0, b0) + 1.f);
    const float ih = fmaxf(0.f, fminf(a3, b3) - fmaxf(a1, b1) + 1.f);
    if (iw <= 0.f || ih <= 0.f) return 0.f;
    const float inter = iw * ih;
    const float u = (a2-a0+1.f)*(a3-a1+1.f) + (b2-b0+1.f)*(b3-b1+1.f) - inter;
    return (u > 0.f) ? (inter / u) : 0.f;
}

// 4x4 Gauss-Jordan inversion on augmented matrix M[4][8] = [A | I].
__device__ void mat4_inv_inplace(float M[4][8]) {
    for (int col = 0; col < 4; col++) {
        int pivot = col;
        float maxv = fabsf(M[col][col]);
        for (int row = col+1; row < 4; row++) {
            float v = fabsf(M[row][col]);
            if (v > maxv) { maxv = v; pivot = row; }
        }
        if (pivot != col) {
            for (int j = 0; j < 8; j++) {
                float t = M[col][j]; M[col][j] = M[pivot][j]; M[pivot][j] = t;
            }
        }
        float inv = 1.f / M[col][col];
        for (int j = 0; j < 8; j++) M[col][j] *= inv;
        for (int row = 0; row < 4; row++) {
            if (row == col) continue;
            float f = M[row][col];
            for (int j = 0; j < 8; j++) M[row][j] -= f * M[col][j];
        }
    }
}

// In-place Kalman predict for one track (xyah + velocity state, 8x8 model).
__device__ void kalman_predict_one(float* m, float* c) {
    const float h = m[3];
    m[0] += m[4]; m[1] += m[5]; m[2] += m[6]; m[3] += m[7];

    const float sp = h / 20.f, sv = h / 160.f;
    const float q[8] = { sp*sp, sp*sp, 1e-4f, sp*sp, sv*sv, sv*sv, 1e-10f, sv*sv };

    float Fc[64];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            Fc[i*8+j] = c[i*8+j] + (i < 4 ? c[(i+4)*8+j] : 0.f);
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            c[i*8+j] = Fc[i*8+j] + (j < 4 ? Fc[i*8+(j+4)] : 0.f) + (i==j ? q[i] : 0.f);
}

// In-place Kalman update for one track. z[4] is measurement in xyah format.
__device__ void kalman_update_one(float* m, float* P, const float z[4]) {
    const float h = m[3];
    const float std_p = h / 20.f;
    const float R[4] = { std_p*std_p, std_p*std_p, 0.01f, std_p*std_p };

    float Saug[4][8];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++)
            Saug[i][j] = P[i*8+j] + (i==j ? R[i] : 0.f);
        for (int j = 0; j < 4; j++)
            Saug[i][4+j] = (i==j) ? 1.f : 0.f;
    }
    mat4_inv_inplace(Saug);

    float K[32] = {};
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 4; k++)
                K[i*4+j] += P[i*8+k] * Saug[k][4+j];

    float innov[4];
    for (int i = 0; i < 4; i++) innov[i] = z[i] - m[i];

    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++)
            m[i] += K[i*4+j] * innov[j];

    float KHP[64] = {};
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            for (int k = 0; k < 4; k++)
                KHP[i*8+j] += K[i*4+k] * P[k*8+j];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            P[i*8+j] -= KHP[i*8+j];
}

} // anonymous namespace

// ============================================================
// Kernels
// ============================================================

// NMS output -> rescaled (x,y,w,h,score,label) objects on device.
// Invalid entries are marked with label == -1.
__global__ void __launch_bounds__(BYTETRACK_CUDA_MAX_THREADS_PER_BLOCK, 4)
nms_to_objects_kernel(
    const float* __restrict__ boxes,
    const float* __restrict__ scores,
    const float* __restrict__ classes,
    int num_det, float scale, float img_w, float img_h,
    float* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_det || scale <= 0.f) return;

    const float left  = boxes[i*4+0], top    = boxes[i*4+1];
    const float right = boxes[i*4+2], bottom = boxes[i*4+3];
    const float score = scores[i];
    const int label   = __float2int_rn(classes[i] + 0.5f);

    if (!isfinite(left) || !isfinite(top) || !isfinite(right) || !isfinite(bottom) ||
        !isfinite(score) || score <= 0.f) {
        for (int k = 0; k < 6; k++) out[i*6+k] = (k==5) ? -1.f : 0.f;
        return;
    }
    const float inv = 1.f / scale;
    const float x1 = clampf(left*inv,  0.f, img_w-1.f), y1 = clampf(top*inv,    0.f, img_h-1.f);
    const float x2 = clampf(right*inv, 0.f, img_w-1.f), y2 = clampf(bottom*inv, 0.f, img_h-1.f);
    if (x2 <= x1 || y2 <= y1) { out[i*6+4]=0.f; out[i*6+5]=-1.f; return; }
    out[i*6+0]=x1; out[i*6+1]=y1; out[i*6+2]=x2-x1; out[i*6+3]=y2-y1;
    out[i*6+4]=score; out[i*6+5]=__int2float_rn(label);
}

// Batch Kalman predict: one thread per track.
__global__ void __launch_bounds__(BYTETRACK_CUDA_MAX_THREADS_PER_BLOCK, 4)
kalman_predict_kernel(float* mean, float* cov, int n_track)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_track) return;
    kalman_predict_one(mean + t*8, cov + t*64);
}

// IoU cost matrix: cost[i,j] = 1 - iou(track_i, det_j).
__global__ void __launch_bounds__(BYTETRACK_CUDA_MAX_THREADS_PER_BLOCK, 4)
iou_cost_matrix_kernel(
    const float* __restrict__ atlbr, const float* __restrict__ btlbr,
    int n_track, int n_det, float* __restrict__ cost)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_track * n_det) return;
    const int i = idx / n_det, j = idx % n_det;
    cost[idx] = 1.f - iou_tlbr(atlbr[i*4+0], atlbr[i*4+1], atlbr[i*4+2], atlbr[i*4+3],
                                 btlbr[j*4+0], btlbr[j*4+1], btlbr[j*4+2], btlbr[j*4+3]);
}

// Extract valid (high-score) detection tlbr from d_objects.
// Writes compact indices back for later kernel lookup.
__global__ void objects_to_tlbr_kernel(
    const float* d_objects, int max_n, float score_thresh,
    float* d_tlbr, int* d_det_idx, int* d_n_valid)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_n) return;
    if (d_objects[i*6+5] < 0.f) return;
    if (d_objects[i*6+4] < score_thresh) return;
    float w = d_objects[i*6+2], h = d_objects[i*6+3];
    if (w <= 0.f || h <= 0.f) return;
    int idx = atomicAdd(d_n_valid, 1);
    float x = d_objects[i*6+0], y = d_objects[i*6+1];
    d_tlbr[idx*4+0] = x;   d_tlbr[idx*4+1] = y;
    d_tlbr[idx*4+2] = x+w; d_tlbr[idx*4+3] = y+h;
    d_det_idx[idx] = i;
}

// Convert track Kalman mean (xyah) -> tlbr.
// Only processes TRACKED and LOST tracks.
__global__ void mean_to_tlbr_kernel(
    const float* d_mean, const char* d_state, const int* d_num_tracks,
    float* d_tlbr, int* d_pool_to_slot, int* d_n_pool)
{
    __shared__ int s_n;
    if (threadIdx.x == 0) s_n = *d_num_tracks;
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s_n) return;
    char s = d_state[i];
    if (s != TRACK_TRACKED && s != TRACK_LOST) return;
    float xc = d_mean[i*8+0], yc = d_mean[i*8+1], a = d_mean[i*8+2], h = d_mean[i*8+3];
    float w = a * h;
    int idx = atomicAdd(d_n_pool, 1);
    d_tlbr[idx*4+0] = xc - w*0.5f; d_tlbr[idx*4+1] = yc - h*0.5f;
    d_tlbr[idx*4+2] = xc + w*0.5f; d_tlbr[idx*4+3] = yc + h*0.5f;
    d_pool_to_slot[idx] = i;
}

// Greedy linear assignment (single-block, single-thread serial).
// shared memory: 2048 bytes for row_used + 2048 bytes for col_used.
__global__ void greedy_lap_kernel(
    const float* d_cost, const int* d_n_pool, const int* d_n_high,
    float thresh, int* d_row_sol, int* d_col_sol)
{
    __shared__ int s_n_pool, s_n_high;
    if (threadIdx.x == 0) { s_n_pool = *d_n_pool; s_n_high = *d_n_high; }
    __syncthreads();
    extern __shared__ char smem[];
    char* row_used = smem;
    char* col_used = smem + 2048;
    const int n_track = s_n_pool, n_det = s_n_high;
    for (int i = threadIdx.x; i < n_track; i += blockDim.x) row_used[i] = 0;
    for (int j = threadIdx.x; j < n_det;  j += blockDim.x) col_used[j] = 0;
    __syncthreads();
    if (threadIdx.x != 0) return;
    for (int i = 0; i < n_track; i++) d_row_sol[i] = -1;
    for (int j = 0; j < n_det;  j++) d_col_sol[j] = -1;
    for (int k = 0; k < n_track; k++) {
        float best = thresh;
        int bi = -1, bj = -1;
        for (int i = 0; i < n_track; i++) {
            if (row_used[i]) continue;
            for (int j = 0; j < n_det; j++) {
                if (col_used[j]) continue;
                float c = d_cost[i * n_det + j];
                if (c < best) { best = c; bi = i; bj = j; }
            }
        }
        if (bi < 0) break;
        d_row_sol[bi] = bj; d_col_sol[bj] = bi;
        row_used[bi] = 1;   col_used[bj] = 1;
    }
}

// Kalman update for matched track-detection pairs.
__global__ void kalman_update_kernel(
    float* d_mean, float* d_cov,
    const int* d_pool_to_slot, const int* d_row_sol, const int* d_n_pool,
    const float* d_objects, const int* d_det_idx)
{
    __shared__ int s_n;
    if (threadIdx.x == 0) s_n = *d_n_pool;
    __syncthreads();
    int pool_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (pool_i >= s_n) return;
    int det_j = d_row_sol[pool_i];
    if (det_j < 0) return;

    int slot = d_pool_to_slot[pool_i];
    int orig  = d_det_idx[det_j];

    float ox = d_objects[orig*6+0], oy = d_objects[orig*6+1];
    float ow = d_objects[orig*6+2], oh = d_objects[orig*6+3];
    float z[4] = { ox + ow*0.5f, oy + oh*0.5f, ow / oh, oh };

    kalman_update_one(d_mean + slot*8, d_cov + slot*64, z);
}

// Update track state: matched -> TRACKED, unmatched -> LOST.
__global__ void update_track_states_kernel(
    char* d_state, char* d_activated, float* d_score, int* d_track_lost_frame,
    const int* d_pool_to_slot, const int* d_row_sol, const int* d_n_pool,
    const float* d_objects, const int* d_det_idx, int frame_id)
{
    __shared__ int s_n;
    if (threadIdx.x == 0) s_n = *d_n_pool;
    __syncthreads();
    int pool_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (pool_i >= s_n) return;
    int slot  = d_pool_to_slot[pool_i];
    int det_j = d_row_sol[pool_i];
    if (det_j >= 0) {
        d_state[slot]           = (char)TRACK_TRACKED;
        d_activated[slot]       = 1;
        int orig = d_det_idx[det_j];
        d_score[slot]           = d_objects[orig*6+4];
        d_track_lost_frame[slot] = 0;
    } else {
        char s = d_state[slot];
        if (s == TRACK_TRACKED || s == TRACK_NEW) {
            d_state[slot]            = (char)TRACK_LOST;
            d_track_lost_frame[slot] = frame_id;
        }
    }
}

// Allocate a new track slot for each unmatched high-score detection.
__global__ void add_new_tracks_kernel(
    const float* d_objects, float high_thresh,
    const int* d_col_sol, const int* d_n_high, const int* d_det_idx,
    float* d_mean, float* d_cov, int* d_id, char* d_state,
    int* d_start, char* d_activated, float* d_score, int* d_track_lost_frame,
    int* d_num_tracks, int* d_next_id, int frame_id, int max_tracks)
{
    __shared__ int s_n;
    if (threadIdx.x == 0) s_n = *d_n_high;
    __syncthreads();
    int compact_j = blockIdx.x * blockDim.x + threadIdx.x;
    if (compact_j >= s_n) return;
    if (d_col_sol[compact_j] >= 0) return;

    int orig  = d_det_idx[compact_j];
    float score = d_objects[orig*6+4];
    if (score < high_thresh) return;

    float x = d_objects[orig*6+0], y = d_objects[orig*6+1];
    float w = d_objects[orig*6+2], h = d_objects[orig*6+3];
    if (w <= 0.f || h <= 0.f) return;

    int slot = atomicAdd(d_num_tracks, 1);
    if (slot >= max_tracks) { atomicAdd(d_num_tracks, -1); return; }
    int id = atomicAdd(d_next_id, 1);

    float xc = x+w*0.5f, yc = y+h*0.5f, a = w/h;
    d_mean[slot*8+0]=xc; d_mean[slot*8+1]=yc; d_mean[slot*8+2]=a;  d_mean[slot*8+3]=h;
    d_mean[slot*8+4]=0.f; d_mean[slot*8+5]=0.f; d_mean[slot*8+6]=0.f; d_mean[slot*8+7]=0.f;

    const float std_pos = (2.f/20.f)*h, std_vel = (10.f/160.f)*h;
    for (int k = 0; k < 64; k++) d_cov[slot*64+k] = 0.f;
    d_cov[slot*64+ 0]=std_pos*std_pos; d_cov[slot*64+ 9]=std_pos*std_pos;
    d_cov[slot*64+18]=1e-4f;           d_cov[slot*64+27]=std_pos*std_pos;
    d_cov[slot*64+36]=std_vel*std_vel; d_cov[slot*64+45]=std_vel*std_vel;
    d_cov[slot*64+54]=1e-10f;          d_cov[slot*64+63]=std_vel*std_vel;

    d_id[slot] = id;
    d_state[slot] = (char)TRACK_TRACKED;
    d_start[slot] = frame_id;
    d_activated[slot] = 1;
    d_score[slot] = score;
    d_track_lost_frame[slot] = 0;
}

// Initialize tracks from high-score detections (first frame or after reset).
__global__ void init_tracks_kernel(
    const float* d_objects, int max_n, float high_thresh,
    float* d_mean, float* d_cov, int* d_id, char* d_state,
    int* d_start, char* d_activated, float* d_score, int* d_track_lost_frame,
    int* d_num_tracks, int* d_next_id, int frame_id, int max_tracks)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_n) return;
    if (d_objects[i*6+5] < 0.f) return;
    float score = d_objects[i*6+4];
    if (score < high_thresh) return;
    float x = d_objects[i*6+0], y = d_objects[i*6+1];
    float w = d_objects[i*6+2], h = d_objects[i*6+3];
    if (w <= 0.f || h <= 0.f) return;

    int slot = atomicAdd(d_num_tracks, 1);
    if (slot >= max_tracks) { atomicAdd(d_num_tracks, -1); return; }
    int id = atomicAdd(d_next_id, 1);

    float xc = x+w*0.5f, yc = y+h*0.5f, a = w/h;
    d_mean[slot*8+0]=xc; d_mean[slot*8+1]=yc; d_mean[slot*8+2]=a;  d_mean[slot*8+3]=h;
    d_mean[slot*8+4]=0.f; d_mean[slot*8+5]=0.f; d_mean[slot*8+6]=0.f; d_mean[slot*8+7]=0.f;

    const float std_pos = (2.f/20.f)*h, std_vel = (10.f/160.f)*h;
    for (int k = 0; k < 64; k++) d_cov[slot*64+k] = 0.f;
    d_cov[slot*64+ 0]=std_pos*std_pos; d_cov[slot*64+ 9]=std_pos*std_pos;
    d_cov[slot*64+18]=1e-4f;           d_cov[slot*64+27]=std_pos*std_pos;
    d_cov[slot*64+36]=std_vel*std_vel; d_cov[slot*64+45]=std_vel*std_vel;
    d_cov[slot*64+54]=1e-10f;          d_cov[slot*64+63]=std_vel*std_vel;

    d_id[slot] = id;
    d_state[slot] = (char)TRACK_TRACKED;
    d_start[slot] = frame_id;
    d_activated[slot] = 1;
    d_score[slot] = score;
    d_track_lost_frame[slot] = 0;
}

// Mark timed-out LOST tracks as REMOVED.
__global__ void remove_old_tracks_kernel(
    char* d_state, const int* d_track_lost_frame,
    const int* d_num_tracks, int frame_id, int max_time_lost)
{
    __shared__ int s_n;
    if (threadIdx.x == 0) s_n = *d_num_tracks;
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s_n) return;
    if (d_state[i] == TRACK_LOST &&
        d_track_lost_frame[i] > 0 &&
        (frame_id - d_track_lost_frame[i]) > max_time_lost) {
        d_state[i] = (char)TRACK_REMOVED;
    }
}

// Collect output: TRACKED + activated tracks -> compact result arrays.
// Each output entry: [id, tlwh[4], score] – 6 floats at d_out_data[idx*6].
__global__ void output_active_kernel(
    const float* d_mean, const float* d_score, const int* d_id,
    const char* d_state, const char* d_activated,
    const int* d_num_tracks,
    int* d_out_id, float* d_out_tlwh, float* d_out_score, int* d_out_count)
{
    __shared__ int s_n;
    if (threadIdx.x == 0) s_n = *d_num_tracks;
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s_n) return;
    if (d_state[i] != TRACK_TRACKED || d_activated[i] == 0) return;
    float xc = d_mean[i*8+0], yc = d_mean[i*8+1], a = d_mean[i*8+2], h = d_mean[i*8+3];
    float w = a * h;
    int idx = atomicAdd(d_out_count, 1);
    d_out_id[idx]        = d_id[i];
    d_out_tlwh[idx*4+0] = xc - w*0.5f;
    d_out_tlwh[idx*4+1] = yc - h*0.5f;
    d_out_tlwh[idx*4+2] = w;
    d_out_tlwh[idx*4+3] = h;
    d_out_score[idx]     = d_score[i];
}

// ============================================================
// Per-instance GPU tracker state
// ============================================================
struct ByteTrackGpuContext
{
    // Persistent track state (lives across frames)
    float* d_track_mean;       // [max_tracks * 8]
    float* d_track_cov;        // [max_tracks * 64]
    int*   d_track_id;
    char*  d_track_state;
    int*   d_track_start;
    char*  d_track_activated;
    float* d_track_score;
    int*   d_track_lost_frame;
    int*   d_num_tracks;
    int*   d_next_id;

    // Per-frame working buffers (reused every frame)
    float* d_objects;       // [max_det * 6]  NMS -> rescaled objects
    float* d_atlbr;         // [max_tracks * 4]  track TLBRs
    float* d_btlbr;         // [max_det * 4]     detection TLBRs
    float* d_cost;          // [max_tracks * max_det]
    int*   d_n_high;
    int*   d_det_idx;       // [max_det]
    int*   d_n_pool;
    int*   d_pool_to_slot;  // [max_tracks]
    int*   d_row_sol;       // [max_tracks]
    int*   d_col_sol;       // [max_det]

    // Output buffers (filled by output_active_kernel, copied to host last)
    int*   d_out_count;
    int*   d_out_id;        // [BYTETRACK_GPU_OUT_MAX]
    float* d_out_tlwh;      // [BYTETRACK_GPU_OUT_MAX * 4]
    float* d_out_score;     // [BYTETRACK_GPU_OUT_MAX]

    int max_tracks;
    int max_det;
};

// ============================================================
// extern "C" API
// ============================================================
extern "C" {

ByteTrackGpuContext* bytetrack_gpu_create(int max_det, int max_tracks)
{
    if (max_det    <= 0) max_det    = 2000;
    if (max_tracks <= 0) max_tracks = 2048;

    auto* ctx = new (std::nothrow) ByteTrackGpuContext();
    if (!ctx) return nullptr;
    memset(ctx, 0, sizeof(*ctx));
    ctx->max_tracks = max_tracks;
    ctx->max_det    = max_det;

    const size_t nt = (size_t)max_tracks;
    const size_t nd = (size_t)max_det;

    auto alloc = [](void** p, size_t bytes) {
        return cudaMalloc(p, bytes) == cudaSuccess;
    };

    bool ok =
        alloc((void**)&ctx->d_track_mean,       nt * 8  * sizeof(float)) &&
        alloc((void**)&ctx->d_track_cov,        nt * 64 * sizeof(float)) &&
        alloc((void**)&ctx->d_track_id,         nt      * sizeof(int))   &&
        alloc((void**)&ctx->d_track_state,      nt)                       &&
        alloc((void**)&ctx->d_track_start,      nt      * sizeof(int))   &&
        alloc((void**)&ctx->d_track_activated,  nt)                       &&
        alloc((void**)&ctx->d_track_score,      nt      * sizeof(float)) &&
        alloc((void**)&ctx->d_track_lost_frame, nt      * sizeof(int))   &&
        alloc((void**)&ctx->d_num_tracks,       sizeof(int))              &&
        alloc((void**)&ctx->d_next_id,          sizeof(int))              &&
        alloc((void**)&ctx->d_objects,          nd * 6  * sizeof(float)) &&
        alloc((void**)&ctx->d_atlbr,            nt * 4  * sizeof(float)) &&
        alloc((void**)&ctx->d_btlbr,            nd * 4  * sizeof(float)) &&
        alloc((void**)&ctx->d_cost,             nt * nd * sizeof(float)) &&
        alloc((void**)&ctx->d_n_high,           sizeof(int))              &&
        alloc((void**)&ctx->d_det_idx,          nd      * sizeof(int))   &&
        alloc((void**)&ctx->d_n_pool,           sizeof(int))              &&
        alloc((void**)&ctx->d_pool_to_slot,     nt      * sizeof(int))   &&
        alloc((void**)&ctx->d_row_sol,          nt      * sizeof(int))   &&
        alloc((void**)&ctx->d_col_sol,          nd      * sizeof(int))   &&
        alloc((void**)&ctx->d_out_count,        sizeof(int))                           &&
        alloc((void**)&ctx->d_out_id,    BYTETRACK_GPU_OUT_MAX * sizeof(int))        &&
        alloc((void**)&ctx->d_out_tlwh,  BYTETRACK_GPU_OUT_MAX * 4 * sizeof(float))  &&
        alloc((void**)&ctx->d_out_score, BYTETRACK_GPU_OUT_MAX * sizeof(float));

    if (!ok) {
        bytetrack_gpu_destroy(ctx);
        return nullptr;
    }

    bytetrack_gpu_reset(ctx);
    return ctx;
}

void bytetrack_gpu_reset(ByteTrackGpuContext* ctx)
{
    if (!ctx || !ctx->d_num_tracks) return;
    cudaMemset(ctx->d_track_state,     0, (size_t)ctx->max_tracks);
    cudaMemset(ctx->d_track_activated, 0, (size_t)ctx->max_tracks);
    int zero = 0, one = 1;
    cudaMemcpy(ctx->d_num_tracks, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_next_id,    &one,  sizeof(int), cudaMemcpyHostToDevice);
}

void bytetrack_gpu_destroy(ByteTrackGpuContext* ctx)
{
    if (!ctx) return;
    auto cfree = [](void* p) { if (p) cudaFree(p); };
    cfree(ctx->d_track_mean);       cfree(ctx->d_track_cov);
    cfree(ctx->d_track_id);         cfree(ctx->d_track_state);
    cfree(ctx->d_track_start);      cfree(ctx->d_track_activated);
    cfree(ctx->d_track_score);      cfree(ctx->d_track_lost_frame);
    cfree(ctx->d_num_tracks);       cfree(ctx->d_next_id);
    cfree(ctx->d_objects);          cfree(ctx->d_atlbr);
    cfree(ctx->d_btlbr);            cfree(ctx->d_cost);
    cfree(ctx->d_n_high);           cfree(ctx->d_det_idx);
    cfree(ctx->d_n_pool);           cfree(ctx->d_pool_to_slot);
    cfree(ctx->d_row_sol);          cfree(ctx->d_col_sol);
    cfree(ctx->d_out_count);        cfree(ctx->d_out_id);
    cfree(ctx->d_out_tlwh);         cfree(ctx->d_out_score);
    delete ctx;
}

int bytetrack_gpu_update(
    ByteTrackGpuContext* ctx,
    const float* d_num_det,
    const float* d_boxes,
    const float* d_scores,
    const float* d_classes,
    float scale, int img_w, int img_h, int frame_id,
    int*   h_out_count,
    int*   h_out_track_id,
    float* h_out_tlwh,
    float* h_out_score,
    int fps, int track_buffer,
    cudaStream_t stream)
{
    if (!ctx || !h_out_count || !h_out_track_id || !h_out_tlwh) return 0;
    *h_out_count = 0;

    // Read num_det from device (one small D2H, mandatory for kernel launch bounds).
    float h_num = 0.f;
    cudaMemcpyAsync(&h_num, d_num_det, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int num_det = std::min((int)(h_num + 0.5f), ctx->max_det);
    if (num_det <= 0) return 0;

    const int B           = BYTETRACK_CUDA_MAX_THREADS_PER_BLOCK;
    const int max_time_lost = (int)((float)fps / 30.f * track_buffer);

    // Step 1: NMS output -> rescaled objects on device.
    nms_to_objects_kernel<<<(num_det+B-1)/B, B, 0, stream>>>(
        d_boxes, d_scores, d_classes, num_det, scale, (float)img_w, (float)img_h,
        ctx->d_objects);

    // Step 2: Filter high-score detections -> compact TLBR + index map.
    cudaMemsetAsync(ctx->d_n_high, 0, sizeof(int), stream);
    objects_to_tlbr_kernel<<<(num_det+B-1)/B, B, 0, stream>>>(
        ctx->d_objects, num_det, 0.5f, ctx->d_btlbr, ctx->d_det_idx, ctx->d_n_high);

    cudaStreamSynchronize(stream);
    int n_high = 0;
    cudaMemcpy(&n_high, ctx->d_n_high, sizeof(int), cudaMemcpyDeviceToHost);

    int n_tracks = 0;
    cudaMemcpy(&n_tracks, ctx->d_num_tracks, sizeof(int), cudaMemcpyDeviceToHost);

    if (n_tracks == 0) {
        // First frame or after reset: initialize tracks directly.
        if (n_high > 0) {
            init_tracks_kernel<<<(num_det+B-1)/B, B, 0, stream>>>(
                ctx->d_objects, num_det, 0.6f,
                ctx->d_track_mean, ctx->d_track_cov, ctx->d_track_id, ctx->d_track_state,
                ctx->d_track_start, ctx->d_track_activated, ctx->d_track_score,
                ctx->d_track_lost_frame, ctx->d_num_tracks, ctx->d_next_id,
                frame_id, ctx->max_tracks);
        }
    } else {
        // Step 3: Kalman predict all TRACKED/LOST tracks.
        kalman_predict_kernel<<<(ctx->max_tracks+B-1)/B, B, 0, stream>>>(
            ctx->d_track_mean, ctx->d_track_cov, n_tracks);

        // Step 4: Convert predicted track means to TLBR for IoU.
        cudaMemsetAsync(ctx->d_n_pool, 0, sizeof(int), stream);
        mean_to_tlbr_kernel<<<(ctx->max_tracks+B-1)/B, B, 0, stream>>>(
            ctx->d_track_mean, ctx->d_track_state, ctx->d_num_tracks,
            ctx->d_atlbr, ctx->d_pool_to_slot, ctx->d_n_pool);

        cudaStreamSynchronize(stream);
        int n_pool = 0;
        cudaMemcpy(&n_pool, ctx->d_n_pool, sizeof(int), cudaMemcpyDeviceToHost);

        if (n_pool > 0 && n_high > 0) {
            // Step 5: IoU cost matrix (entirely on device).
            iou_cost_matrix_kernel<<<(n_pool*n_high+B-1)/B, B, 0, stream>>>(
                ctx->d_atlbr, ctx->d_btlbr, n_pool, n_high, ctx->d_cost);

            // Step 6: Greedy linear assignment (single-block serial kernel).
            greedy_lap_kernel<<<1, 256, 2*2048, stream>>>(
                ctx->d_cost, ctx->d_n_pool, ctx->d_n_high, 0.8f,
                ctx->d_row_sol, ctx->d_col_sol);

            // Step 7: Kalman update for matched track-detection pairs.
            kalman_update_kernel<<<(n_pool+B-1)/B, B, 0, stream>>>(
                ctx->d_track_mean, ctx->d_track_cov,
                ctx->d_pool_to_slot, ctx->d_row_sol, ctx->d_n_pool,
                ctx->d_objects, ctx->d_det_idx);

            // Step 8: Update states (matched->TRACKED, unmatched->LOST).
            update_track_states_kernel<<<(n_pool+B-1)/B, B, 0, stream>>>(
                ctx->d_track_state, ctx->d_track_activated, ctx->d_track_score,
                ctx->d_track_lost_frame,
                ctx->d_pool_to_slot, ctx->d_row_sol, ctx->d_n_pool,
                ctx->d_objects, ctx->d_det_idx, frame_id);

            // Step 9: Unmatched high-score detections -> new tracks.
            add_new_tracks_kernel<<<(n_high+B-1)/B, B, 0, stream>>>(
                ctx->d_objects, 0.6f,
                ctx->d_col_sol, ctx->d_n_high, ctx->d_det_idx,
                ctx->d_track_mean, ctx->d_track_cov, ctx->d_track_id, ctx->d_track_state,
                ctx->d_track_start, ctx->d_track_activated, ctx->d_track_score,
                ctx->d_track_lost_frame,
                ctx->d_num_tracks, ctx->d_next_id, frame_id, ctx->max_tracks);

        } else if (n_pool == 0 && n_high > 0) {
            // No existing tracks, initialize from detections.
            init_tracks_kernel<<<(num_det+B-1)/B, B, 0, stream>>>(
                ctx->d_objects, num_det, 0.6f,
                ctx->d_track_mean, ctx->d_track_cov, ctx->d_track_id, ctx->d_track_state,
                ctx->d_track_start, ctx->d_track_activated, ctx->d_track_score,
                ctx->d_track_lost_frame, ctx->d_num_tracks, ctx->d_next_id,
                frame_id, ctx->max_tracks);

        } else if (n_pool > 0 && n_high == 0) {
            // No detections: all active tracks go to LOST.
            cudaMemset(ctx->d_row_sol, -1, (size_t)n_pool * sizeof(int));
            update_track_states_kernel<<<(n_pool+B-1)/B, B, 0, stream>>>(
                ctx->d_track_state, ctx->d_track_activated, ctx->d_track_score,
                ctx->d_track_lost_frame,
                ctx->d_pool_to_slot, ctx->d_row_sol, ctx->d_n_pool,
                ctx->d_objects, ctx->d_det_idx, frame_id);
        }

        // Step 10: Age out timed-out LOST tracks.
        if (max_time_lost > 0) {
            remove_old_tracks_kernel<<<(ctx->max_tracks+B-1)/B, B, 0, stream>>>(
                ctx->d_track_state, ctx->d_track_lost_frame, ctx->d_num_tracks,
                frame_id, max_time_lost);
        }
    }

    // Step 11: Gather active tracks into output buffers (still on device).
    cudaMemsetAsync(ctx->d_out_count, 0, sizeof(int), stream);
    output_active_kernel<<<(ctx->max_tracks+B-1)/B, B, 0, stream>>>(
        ctx->d_track_mean, ctx->d_track_score, ctx->d_track_id,
        ctx->d_track_state, ctx->d_track_activated, ctx->d_num_tracks,
        ctx->d_out_id, ctx->d_out_tlwh, ctx->d_out_score, ctx->d_out_count);

    // Single D2H transfer: only the compact result list leaves the GPU.
    cudaStreamSynchronize(stream);
    int out_count = 0;
    cudaMemcpy(&out_count, ctx->d_out_count, sizeof(int), cudaMemcpyDeviceToHost);
    *h_out_count = out_count;
    if (out_count > 0) {
        cudaMemcpy(h_out_track_id, ctx->d_out_id,    (size_t)out_count * sizeof(int),       cudaMemcpyDeviceToHost);
        cudaMemcpy(h_out_tlwh,     ctx->d_out_tlwh,  (size_t)out_count * 4 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_out_score,    ctx->d_out_score,  (size_t)out_count * sizeof(float),     cudaMemcpyDeviceToHost);
    }
    return out_count;
}

} // extern "C"
