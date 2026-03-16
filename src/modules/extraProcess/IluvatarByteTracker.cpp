#include "IluvatarByteTracker.h"
#include "bytetrack_cuda.h"

#include <vector>

// ============================================================
// Private implementation (holds GPU context + output scratch)
// ============================================================
struct ByteTrackerImpl
{
    ByteTrackGpuContext* gpu_ctx     = nullptr;
    int                  frame_id    = 0;
    int                  frame_rate;
    int                  track_buffer;

    // Host-side output scratch (reused each frame, avoids repeated allocation).
    std::vector<int>   h_out_track_id;
    std::vector<float> h_out_tlwh;
    std::vector<float> h_out_score;
    int                h_out_count = 0;
};

// ============================================================
// ByteTracker public API
// ============================================================
ByteTracker::ByteTracker(int frame_rate, int track_buffer, int max_det, int max_tracks)
{
    p_impl               = new ByteTrackerImpl();
    p_impl->frame_rate   = frame_rate;
    p_impl->track_buffer = track_buffer;
    p_impl->h_out_track_id.resize(BYTETRACK_GPU_OUT_MAX);
    p_impl->h_out_tlwh.resize(static_cast<size_t>(BYTETRACK_GPU_OUT_MAX) * 4);
    p_impl->h_out_score.resize(BYTETRACK_GPU_OUT_MAX);
    p_impl->gpu_ctx = bytetrack_gpu_create(max_det, max_tracks);
}

ByteTracker::~ByteTracker()
{
    if (p_impl)
    {
        bytetrack_gpu_destroy(p_impl->gpu_ctx);
        delete p_impl;
        p_impl = nullptr;
    }
}

void ByteTracker::Reset()
{
    if (p_impl)
    {
        p_impl->frame_id = 0;
        bytetrack_gpu_reset(p_impl->gpu_ctx);
    }
}

std::vector<TrackResult> ByteTracker::Update(const float* d_num_det,
                                              const float* d_boxes,
                                              const float* d_scores,
                                              const float* d_classes,
                                              float        scale,
                                              int          img_w,
                                              int          img_h,
                                              cudaStream_t stream)
{
    p_impl->frame_id++;

    bytetrack_gpu_update(
        p_impl->gpu_ctx,
        d_num_det, d_boxes, d_scores, d_classes,
        scale, img_w, img_h, p_impl->frame_id,
        &p_impl->h_out_count,
        p_impl->h_out_track_id.data(),
        p_impl->h_out_tlwh.data(),
        p_impl->h_out_score.data(),
        p_impl->frame_rate, p_impl->track_buffer,
        stream);

    // Convert compact GPU output to TrackResult vector.
    std::vector<TrackResult> results;
    results.reserve(p_impl->h_out_count);
    for (int i = 0; i < p_impl->h_out_count; ++i)
    {
        const float* t = p_impl->h_out_tlwh.data() + i * 4;
        const int w = static_cast<int>(t[2]);
        const int h = static_cast<int>(t[3]);
        if (w <= 0 || h <= 0) continue;

        TrackResult res;
        res.track_id    = p_impl->h_out_track_id[i];
        res.rect.x      = static_cast<int>(t[0]);
        res.rect.y      = static_cast<int>(t[1]);
        res.rect.width  = w;
        res.rect.height = h;
        res.score       = p_impl->h_out_score[i];
        res.label       = 0;
        results.push_back(res);
    }
    return results;
}
