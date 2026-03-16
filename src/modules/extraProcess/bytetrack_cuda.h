#pragma once

#include <cuda_runtime.h>

// Maximum number of tracks returned per frame.
#define BYTETRACK_GPU_OUT_MAX 2048

// Opaque per-instance GPU tracker state.
// Each video stream should hold its own context so that their track states
// remain independent even when processed on the same GPU.
struct ByteTrackGpuContext;

#ifdef __cplusplus
extern "C" {
#endif

// Allocate and initialize GPU memory for one tracker instance.
// max_det   : maximum detections per frame (e.g. 2000)
// max_tracks: maximum concurrent tracks    (e.g. 2048)
ByteTrackGpuContext* bytetrack_gpu_create(int max_det, int max_tracks);

// Reset track state (clear all tracks), keeping GPU memory allocated.
// Call this when a video stream reconnects / restarts.
void bytetrack_gpu_reset(ByteTrackGpuContext* ctx);

// Free all GPU memory for this instance.
void bytetrack_gpu_destroy(ByteTrackGpuContext* ctx);

// Run one frame of the full GPU ByteTrack pipeline.
//
// All intermediate data (Kalman predict, IoU cost matrix, assignment, state
// update) stay on the GPU.  Only the final compact result list is copied
// back to host at the very end.
//
// Inputs (all device pointers, direct from inference engine output):
//   d_num_det  [1]          – number of valid detections (float on device)
//   d_boxes    [max_det*4]  – detection boxes  (ltrb in model-input space)
//   d_scores   [max_det]    – detection scores
//   d_classes  [max_det]    – detection class ids
//   scale      – letterbox ratio (model_side / original_side)
//   img_w/h    – original image resolution
//   frame_id   – monotonically increasing frame counter (managed by caller)
//   fps, track_buffer – same semantics as CPU BYTETracker
//   stream     – CUDA stream; pass the same stream used for inference to
//                avoid an explicit sync between infer and tracking
//
// Outputs (host buffers, caller-allocated):
//   h_out_count    – number of active tracks this frame
//   h_out_track_id [BYTETRACK_GPU_OUT_MAX]    – track IDs
//   h_out_tlwh     [BYTETRACK_GPU_OUT_MAX * 4] – bounding boxes (x,y,w,h)
//
// Returns h_out_count.
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
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
