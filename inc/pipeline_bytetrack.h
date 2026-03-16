#pragma once

#include "pipeline_one_stage.h"
#include "IluvatarByteTracker.h"
#include <mutex>

struct ByteTrackResult
{
    ViDecSurfaceCudaBuff  oriSurface;
    std::vector<Rect>     rects;
    std::vector<RectInfo> rectsInfo;
    std::vector<int>      trackIds;
};

// Snapshot of the latest tracking result for one stream.
// Updated by IxrtInfer after every tracked frame; read by VideoEncode for every raw frame.
struct StreamTrackState
{
    std::vector<Rect>     rects;
    std::vector<RectInfo> rectsInfo;
    std::vector<int>      trackIds;
};

struct ByteTrackPreSurfaces
{
    SurfaceCudaBuff                   preSurface;
    std::vector<ViDecSurfaceCudaBuff> oriSurfaces;
};

class ByteTrackModelProcessor : public ModelProcessor
{
private:
    ProcessQueue<ViDecSurfaceCudaBuff>* _video_queue        = nullptr;
    ProcessQueue<ViDecSurfaceCudaBuff>* _jpeg_queue         = nullptr;
    ProcessQueue<ViDecSurfaceCudaBuff>* _encode_raw_queue   = nullptr; // all frames → VideoEncode
    ProcessQueue<ByteTrackResult>*      _track_res_queue    = nullptr;
    ProcessQueue<ByteTrackPreSurfaces>* _m_preprocess_queue = nullptr;
    DetectPreprocessor*                 _m_bytetrack_preprocessor = nullptr;
    std::map<size_t, ByteTracker*>      _trackers;
    int                                 _frame_rate    = 30;
    // Per-stream track rate (keyed by RTSP-URL hash).
    // Value N means: run inference on 1 out of every N decoded frames.
    // Populated from RtspUrlParams.rate at construction time.
    std::map<size_t, int>               _track_rate_map;
    // Per-stream decoded-frame counter used by CudaPreProcess for rate filtering.
    std::map<size_t, int>               _preproc_counters;
    void*                               _plugin_handle = nullptr;

    // Latest tracking state per stream (keyed by RTSP-URL hash).
    // Written by IxrtInfer; read by VideoEncode via GetLatestTracks().
    mutable std::mutex                       _latest_tracks_mtx;
    std::map<size_t, StreamTrackState>       _latest_tracks;

    virtual int  InitPreProcess() override;
    virtual int  InitIxrtIfner() override;
    virtual void IxrtInfer() override;
    virtual void CudaPreProcess() override;

public:
    ByteTrackModelProcessor(CUcontext                           cu_context,
                            int                                 resize_h,
                            int                                 resize_w,
                            int                                 pre_maxBatch,
                            int                                 pre_qsz,
                            int                                 infer_maxBatch,
                            const std::vector<std::string>&     custom_outputs,
                            const std::string&                  onnx_file,
                            const std::string&                  engine_file,
                            ProcessQueue<ViDecSurfaceCudaBuff>* video_queue,
                            ProcessQueue<ViDecSurfaceCudaBuff>* jpeg_queue,
                            ProcessQueue<ViDecSurfaceCudaBuff>* encode_raw_queue,
                            ProcessQueue<ByteTrackResult>*      track_res_queue,
                            int                                 frame_rate,
                            std::map<size_t, int>               track_rate_map);

    ~ByteTrackModelProcessor() override;

    // Thread-safe read of the latest tracking result for a given stream.
    StreamTrackState GetLatestTracks(size_t stream_hash) const
    {
        std::lock_guard<std::mutex> lk(_latest_tracks_mtx);
        auto it = _latest_tracks.find(stream_hash);
        return it != _latest_tracks.end() ? it->second : StreamTrackState{};
    }

    virtual size_t GetPreQWaitNum() override { return _m_preprocess_queue->size(); }
    virtual size_t GetPreQDropNum() override { return _m_preprocess_queue->drops_count(); }
};

class PipeLineProcessorByteTrack
{
private:
    int       _dev_id;
    CUcontext _cu_context;

    int                                  _dec_qsz = 0;
    std::vector<RtspUrlParams>           _rtsp_sources;
    std::map<int, VideoStreamProcessor*> _video_processores;
    std::map<int, int>                   _video_reset_count;
    ProcessQueue<ViDecSurfaceCudaBuff>*  _video_queue = nullptr;

    int InitVideoStreams(const int dev_id);

    std::atomic<size_t>                 _p_jpegEncodeFrames = 0;
    int                                 _p_jpeg_maxBatch    = 0;
    int                                 _jpeg_qsz           = 0;
    IluvatarJpegCodec*                  _jpeg_encoder       = nullptr;
    ProcessQueue<ViDecSurfaceCudaBuff>* _jpeg_queue         = nullptr;
    TaskThread                          _p_thread_jpeg;

    int JpegEncode();

    ByteTrackModelProcessor*       _bytetrack_processor = nullptr;
    ProcessQueue<ByteTrackResult>* _bytetrack_res_queue = nullptr;

    // All decoded frames (no rate filtering) for the CPU video encoder.
    ProcessQueue<ViDecSurfaceCudaBuff>* _encode_raw_queue = nullptr;

    // CPU video encoding with drawn track boxes
    int                      _frame_rate        = 30;
    std::string              _encode_output_dir;
    std::atomic<size_t>      _p_encodeFrames    = 0;
    TaskThread               _p_thread_encode;

    int VideoEncode();

public:
    // encode_output_dir: directory for output .mp4 files, one per RTSP stream.
    //   Pass "" to disable encoding.
    PipeLineProcessorByteTrack(CUcontext                         cu_context,
                               const int                         dev_id,
                               const int                         jpeg_maxBatch,
                               const int                         jpeg_qsz,
                               const int                         dec_qsz,
                               const std::vector<RtspUrlParams>& rtsp_sources,
                               const ModelParams&                bytetrack_params,
                               int                               frame_rate = 30,
                               const std::string&                encode_output_dir = "./track_output");

    size_t GetDecFrameNumber()  { return _video_queue->count(); }
    size_t GetDecQWaitNum()     { return _video_queue->size(); }
    size_t GetDecQDropNum()     { return _video_queue->drops_count(); }

    size_t GetJpegFrameNumber() { return _p_jpegEncodeFrames.load(); }
    size_t GetJpegQWaitNum()    { return _jpeg_queue->size(); }
    size_t GetJpegQDropNum()    { return _jpeg_queue->drops_count(); }

    size_t GetByteTrackPreFrameNumber()   { return _bytetrack_processor->GetPreFrameNumber(); }
    size_t GetByteTrackInferFrameNumber() { return _bytetrack_processor->GetInferFrameNumber(); }
    size_t GetByteTrackPreQWaitNum()      { return _bytetrack_processor->GetPreQWaitNum(); }
    size_t GetByteTrackPreQDropNum()      { return _bytetrack_processor->GetPreQDropNum(); }

    size_t GetByteTrackResQWaitNum()    { return _bytetrack_res_queue->size(); }
    size_t GetByteTrackResQDropNum()    { return _bytetrack_res_queue->drops_count(); }
    size_t GetEncodeRawQWaitNum()       { return _encode_raw_queue->size(); }
    size_t GetEncodeRawQDropNum()       { return _encode_raw_queue->drops_count(); }

    size_t GetEncodeFrameNumber() { return _p_encodeFrames.load(); }

    std::map<int, int>                 ChcekDecoderStautsAndRestart();
    std::vector<DecoderProcessorStats> CollectVideoProcessorStats();
    int                                CollectVideoProcessorNumbers() { return _video_processores.size(); }

    int StartPipeline();
    ~PipeLineProcessorByteTrack();
};
