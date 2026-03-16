#pragma once

#include "IluvatarPreprocess.h"

#include <cuda_runtime.h>
#include <vector>

// Per-track result returned after each Update() call.
// Only host-side data – all GPU computation has already completed.
struct TrackResult
{
    int   track_id;
    Rect  rect;   // x, y, width, height in original image coordinates
    float score;
    int   label;
};

// GPU-accelerated multi-object tracker (ByteTrack algorithm).
//
// All intermediate data (Kalman filter state, IoU cost matrix, assignment)
// stays on the GPU across the lifetime of the object.
// The only D2H transfer happens at the end of each Update() call when the
// compact result list (track_id + tlwh per active track) is copied back.
//
// Each instance holds independent GPU state, so multiple streams can each
// own a ByteTracker without interfering with each other.
//
// Usage:
//   ByteTracker tracker(fps, track_buffer, max_det);
//
//   // Inside inference loop – pass device output pointers directly:
//   std::vector<TrackResult> results = tracker.Update(
//       d_num_det, d_boxes, d_scores, d_classes,
//       scale, img_w, img_h, infer_stream);
class ByteTracker
{
public:
    ByteTracker()                            = delete;
    ByteTracker(const ByteTracker&)          = delete;
    ByteTracker& operator=(const ByteTracker&) = delete;

    // frame_rate   – source video FPS (used to compute max_time_lost)
    // track_buffer – frames to keep a lost track before removing it
    // max_det      – maximum detections per frame expected from the model
    // max_tracks   – upper bound on concurrent tracks
    explicit ByteTracker(int frame_rate   = 30,
                         int track_buffer = 30,
                         int max_det      = 2000,
                         int max_tracks   = 2048);
    ~ByteTracker();

    // Feed one frame of inference output (all device pointers).
    //   d_num_det  – device scalar: number of valid detections
    //   d_boxes    – device array [max_det * 4]: ltrb in model-input space
    //   d_scores   – device array [max_det]
    //   d_classes  – device array [max_det]
    //   scale      – letterbox ratio: model_side / original_side
    //   img_w/h    – original image resolution
    //   stream     – CUDA stream; use the same stream as inference to avoid
    //                stalling the GPU between inference and tracking
    std::vector<TrackResult> Update(const float* d_num_det,
                                    const float* d_boxes,
                                    const float* d_scores,
                                    const float* d_classes,
                                    float        scale,
                                    int          img_w,
                                    int          img_h,
                                    cudaStream_t stream = nullptr);

    // Clear all track state (e.g. on stream reconnect).
    void Reset();

private:
    struct ByteTrackerImpl* p_impl;
};
