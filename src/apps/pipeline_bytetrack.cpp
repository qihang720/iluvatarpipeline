#include "pipeline_bytetrack.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;

// Generate a visually distinct BGR color for each track ID.
static cv::Scalar track_color(int track_id)
{
    uint32_t h = static_cast<uint32_t>(track_id) * 2654435761u;
    return cv::Scalar(h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF);
}

namespace {

constexpr const char* kInputBlobName       = "images";
constexpr const char* kOutNumDetections    = "num_detections";
constexpr const char* kOutDetectionBoxes   = "detection_boxes";
constexpr const char* kOutDetectionScores  = "detection_scores";
constexpr const char* kOutDetectionClasses = "detection_classes";
constexpr const char* kDefaultIxrtPluginSo =
    "/home/qihang.zhang/code/sw_code/ixrt/third_party/ixrt-oss/build_plugin/lib/liboss_ixrt_plugin.so";

bool file_exists(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

static int AddVideoStream(std::map<int, VideoStreamProcessor*>& list,
                          ProcessQueue<ViDecSurfaceCudaBuff>*   queue,
                          CUcontext cu_context, int dev,
                          const RtspUrlParams& params, int id)
{
    size_t tries = 0;
    while (tries <= 100) {
        try {
            // Always decode every frame (rate=-1 → no decoder-level skipping)
            // so the video encoder sees the full stream.
            // Inference-level skipping is handled by track_rate in CudaPreProcess.
            list[id] = new VideoStreamProcessor(cu_context, dev, params.rtsp_url, queue, id, -1);
            break;
        } catch (const std::exception& e) {
            delete list[id]; list[id] = nullptr;
            ++tries;
            std::cerr << e.what() << " retry " << tries << '\n';
        }
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }
    return tries > 100 ? -1 : 0;
}

static int RestartVideoStream(std::map<int, VideoStreamProcessor*>& list,
                               ProcessQueue<ViDecSurfaceCudaBuff>*   queue,
                               CUcontext cu_context, int dev,
                               const RtspUrlParams& params, int id)
{
    delete list[id]; list[id] = nullptr;
    int ret = AddVideoStream(list, queue, cu_context, dev, params, id);
    if (ret < 0)
        logger->warn("[{} {}]: Restart id {} failed: {}", __FUNCTION__, __LINE__, id, params.rtsp_url);
    return ret;
}

} // namespace

// ============================================================
// ByteTrackModelProcessor
// ============================================================
ByteTrackModelProcessor::ByteTrackModelProcessor(
    CUcontext                           cu_context,
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
    std::map<size_t, int>               track_rate_map)
    : ModelProcessor(cu_context, resize_h, resize_w, pre_maxBatch, pre_qsz, infer_maxBatch,
                     custom_outputs, onnx_file, engine_file)
    , _video_queue(video_queue)
    , _jpeg_queue(jpeg_queue)
    , _encode_raw_queue(encode_raw_queue)
    , _track_res_queue(track_res_queue)
    , _frame_rate(frame_rate)
    , _track_rate_map(std::move(track_rate_map))
{
    InitModelHandle();
}

ByteTrackModelProcessor::~ByteTrackModelProcessor()
{
    for (auto& kv : _trackers) delete kv.second;
    _trackers.clear();

    delete _m_bytetrack_preprocessor;
    _m_bytetrack_preprocessor = nullptr;

    delete _m_infer_model;
    _m_infer_model = nullptr;

    delete _m_preprocess_queue;
    _m_preprocess_queue = nullptr;

    if (_plugin_handle) { dlclose(_plugin_handle); _plugin_handle = nullptr; }
}

int ByteTrackModelProcessor::InitPreProcess()
{
    _m_PreproFrames.store(0);
    _m_preprocess_queue       = new ProcessQueue<ByteTrackPreSurfaces>(_m_pre_qsz);
    _m_bytetrack_preprocessor = new DetectPreprocessor(_m_resize_h, _m_resize_w, _m_cu_context, _m_pre_maxBatch);
    return 0;
}

int ByteTrackModelProcessor::InitIxrtIfner()
{
    _m_InferFrames.store(0);

    // Load optional IxRT plugin.
    std::string plugin_lib;
    const char* env = std::getenv("IXRT_PLUGIN_SO");
    if (env) plugin_lib = env;
    if (plugin_lib.empty() && file_exists(kDefaultIxrtPluginSo))   plugin_lib = kDefaultIxrtPluginSo;
    if (plugin_lib.empty() && file_exists("/usr/local/corex/lib/liboss_ixrt_plugin.so"))
        plugin_lib = "/usr/local/corex/lib/liboss_ixrt_plugin.so";

    if (!plugin_lib.empty()) {
        dlerror();
        _plugin_handle = dlopen(plugin_lib.c_str(), RTLD_LAZY | RTLD_GLOBAL);
        if (!_plugin_handle) {
            logger->error("[{} {}]: load plugin failed: {}", __FUNCTION__, __LINE__, dlerror());
            return -1;
        }
        logger->info("[{} {}]: load plugin: {}", __FUNCTION__, __LINE__, plugin_lib);
    }

    _m_infer_model = new Trt();
    _m_infer_model->mBatchSize = _m_infer_maxBatch;
    _m_infer_model->CreateEngine(_m_onnx_file, _m_engine_file, _m_model_outputs, _m_infer_maxBatch);

    const int input_index = _m_infer_model->mEngine->getBindingIndex(kInputBlobName);
    _m_model_height = _m_infer_model->GetBindingDimsVec(input_index)[2];
    _m_model_width  = _m_infer_model->GetBindingDimsVec(input_index)[3];
    return 0;
}

void ByteTrackModelProcessor::CudaPreProcess()
{
    CudaCtxPush ctxPush(_m_cu_context);
    _m_thread_preprocess.set_status(false);
    while (!_m_thread_preprocess.get_status())
    {
        // Use condition-variable based wait: wakes up immediately when frames
        // arrive instead of sleeping a fixed 20 ms poll interval.
        auto oriSurfaces = _video_queue->get(
            static_cast<size_t>(_m_pre_maxBatch), std::chrono::milliseconds(100)).get();
        if (oriSurfaces.empty()) continue;

        // All frames → JPEG encode queue and raw encode queue (full video stream).
        for (const auto& s : oriSurfaces)
        {
            _jpeg_queue->put(s);
            _encode_raw_queue->put(s);
        }

        // Rate filter: collect only the frames that should be tracked this round.
        // The rate is looked up per-stream from _track_rate_map (derived from
        // RtspUrlParams.rate). Streams absent from the map default to 1 (every frame).
        std::vector<ViDecSurfaceCudaBuff> track_batch;
        track_batch.reserve(oriSurfaces.size());
        for (const auto& s : oriSurfaces)
        {
            const size_t hash = std::hash<std::string>{}(s.GetRtspInfo());
            int& cnt = _preproc_counters[hash];
            ++cnt;
            const auto it = _track_rate_map.find(hash);
            const int  rate = (it != _track_rate_map.end() && it->second > 1) ? it->second : 1;
            if (rate <= 1 || cnt % rate == 1)
                track_batch.push_back(s);
        }

        if (track_batch.empty())
        {
            _m_PreproFrames += oriSurfaces.size();
            continue;
        }

        SurfaceCudaBuff pre = _m_bytetrack_preprocessor->Process(track_batch);
        if (pre.Empty() || pre.GetWidth() == 0 || pre.GetHeight() == 0) {
            logger->warn("[{} {}]: preprocess output empty", __FUNCTION__, __LINE__);
            continue;
        }

        _m_preprocess_queue->put({pre, track_batch});
        _m_PreproFrames += oriSurfaces.size();
    }
}

void ByteTrackModelProcessor::IxrtInfer()
{
    CudaCtxPush ctxPush(_m_cu_context);
    _m_thread_infer.set_status(false);

    // Resolve binding indices once.
    const int idx_input   = _m_infer_model->mEngine->getBindingIndex(kInputBlobName);
    const int idx_num     = _m_infer_model->mEngine->getBindingIndex(kOutNumDetections);
    const int idx_boxes   = _m_infer_model->mEngine->getBindingIndex(kOutDetectionBoxes);
    const int idx_scores  = _m_infer_model->mEngine->getBindingIndex(kOutDetectionScores);
    const int idx_classes = _m_infer_model->mEngine->getBindingIndex(kOutDetectionClasses);
    if (idx_input < 0 || idx_num < 0 || idx_boxes < 0 || idx_scores < 0 || idx_classes < 0) {
        logger->error("[{} {}]: engine binding names mismatch", __FUNCTION__, __LINE__);
        return;
    }

    // Per-image strides in the engine output buffers (element count, not bytes).
    const size_t num_stride     = _m_infer_model->mBindingSize[idx_num]     / sizeof(float);
    const size_t boxes_stride   = _m_infer_model->mBindingSize[idx_boxes]   / sizeof(float);
    const size_t scores_stride  = _m_infer_model->mBindingSize[idx_scores]  / sizeof(float);
    const size_t classes_stride = _m_infer_model->mBindingSize[idx_classes] / sizeof(float);

    while (!_m_thread_infer.get_status())
    {
        // Condition-variable based wait: wakes immediately when a batch is ready.
        auto preSurf = _m_preprocess_queue->get(std::chrono::milliseconds(100)).get();
        if (preSurf.preSurface.Empty()) continue;  // timeout or empty batch

        const int batch = preSurf.preSurface.GetBatch();
        if (static_cast<int>(preSurf.oriSurfaces.size()) != batch) {
            logger->error("[{} {}]: batch size mismatch: ori={} pre={}",
                          __FUNCTION__, __LINE__, preSurf.oriSurfaces.size(), batch);
            continue;
        }

        // Guard: batch must not exceed the engine's configured max batch.
        // If it does, the engine buffers are undersized → CUDA invalid-value error.
        // This happens when infer_maxBatch in JSON was increased but the engine
        // was not yet rebuilt with the matching MAX_BATCH.  Drop the extra frames
        // and log an error so the misconfiguration is immediately visible.
        if (batch > _m_infer_maxBatch) {
            logger->error("[{} {}]: batch={} exceeds engine infer_maxBatch={}, dropping batch. "
                          "Rebuild the engine with MAX_BATCH>={} via models/bytetrack/convert.sh",
                          __FUNCTION__, __LINE__, batch, _m_infer_maxBatch, batch);
            continue;
        }

        // Build binding pointer array for the engine.
        std::vector<void*> buffers(_m_infer_model->nbBindings);
        for (int i = 0; i < _m_infer_model->nbBindings; ++i)
            buffers[i] = _m_infer_model->GetBindingPtr(i);

        // Copy preprocessed frames to engine input (device -> device).
        checkCudaErrors(cudaMemcpyAsync(
            buffers[idx_input],
            reinterpret_cast<void*>(preSurf.preSurface.GetGpuMem()),
            _m_infer_model->mBindingSize[idx_input] * batch,
            cudaMemcpyDeviceToDevice, _m_infer_cudastream));

        // Run inference – all outputs stay on device.
        _m_infer_model->Forward(buffers.data(), batch, _m_infer_cudastream);
        _m_InferFrames += batch;

        // For each image in the batch, call the GPU tracker directly with
        // device output pointers – no intermediate CPU copy needed.
        for (int i = 0; i < batch; ++i)
        {
            const ViDecSurfaceCudaBuff& ori = preSurf.oriSurfaces[i];
            const float ratio_w = static_cast<float>(_m_model_width)  / static_cast<float>(ori.GetWidth());
            const float ratio_h = static_cast<float>(_m_model_height) / static_cast<float>(ori.GetHeight());
            const float scale   = std::min(ratio_w, ratio_h);

            // Device pointers for image i inside the batched output buffers.
            const float* d_num_det  = static_cast<const float*>(buffers[idx_num])     + i * num_stride;
            const float* d_boxes    = static_cast<const float*>(buffers[idx_boxes])   + i * boxes_stride;
            const float* d_scores   = static_cast<const float*>(buffers[idx_scores])  + i * scores_stride;
            const float* d_classes  = static_cast<const float*>(buffers[idx_classes]) + i * classes_stride;

            // Lazily create one ByteTracker per RTSP stream.
            const size_t hash = std::hash<std::string>{}(ori.GetRtspInfo());
            if (_trackers.find(hash) == _trackers.end())
                _trackers[hash] = new ByteTracker(_frame_rate, 30);

            // Update: device pointers go in, TrackResult vector comes out.
            // This is the only D2H transfer – the compact result list at the end.
            std::vector<TrackResult> results = _trackers[hash]->Update(
                d_num_det, d_boxes, d_scores, d_classes,
                scale, ori.GetWidth(), ori.GetHeight(), _m_infer_cudastream);

            // Pack into pipeline result.
            std::vector<Rect>     rects;
            std::vector<RectInfo> rects_info;
            std::vector<int>      track_ids;
            rects.reserve(results.size());
            rects_info.reserve(results.size());
            track_ids.reserve(results.size());

            for (const auto& r : results) {
                track_ids.push_back(r.track_id);
                rects.push_back(r.rect);
                rects_info.push_back({r.label, r.score});
            }

            // Update shared latest-tracks map so VideoEncode can draw boxes on
            // every raw frame (including non-tracked intermediate frames).
            {
                std::lock_guard<std::mutex> lk(_latest_tracks_mtx);
                _latest_tracks[hash] = {rects, rects_info, track_ids};
            }

            // Also publish to the result queue (for external monitoring / stats).
            _track_res_queue->put({ori, rects, rects_info, track_ids});
        }
    }
}

// ============================================================
// PipeLineProcessorByteTrack
// ============================================================
PipeLineProcessorByteTrack::PipeLineProcessorByteTrack(
    CUcontext                         cu_context,
    const int                         dev_id,
    const int                         jpeg_maxBatch,
    const int                         jpeg_qsz,
    const int                         dec_qsz,
    const std::vector<RtspUrlParams>& rtsp_sources,
    const ModelParams&                bytetrack_params,
    int                               frame_rate,
    const std::string&                encode_output_dir)
    : _dev_id(dev_id)
    , _cu_context(cu_context)
    , _dec_qsz(dec_qsz)
    , _rtsp_sources(rtsp_sources)
    , _p_jpeg_maxBatch(jpeg_maxBatch)
    , _jpeg_qsz(jpeg_qsz)
    , _frame_rate(frame_rate)
    , _encode_output_dir(encode_output_dir)
{
    CudaCtxPush ctxPush(_cu_context);
    InitVideoStreams(dev_id);

    _jpeg_queue          = new ProcessQueue<ViDecSurfaceCudaBuff>(jpeg_qsz);
    _jpeg_encoder        = new IluvatarJpegCodec(dev_id, jpeg_maxBatch, false, true);
    // The result queue only needs to hold a small number of entries.
    // In the test binary nobody consumes this queue beyond reading its stats;
    // keeping it tiny prevents accumulation of GPU-memory-holding ByteTrackResult
    // objects that would otherwise pile up until the queue capacity is reached.
    const int res_qsz = std::max(static_cast<int>(_rtsp_sources.size()) * 4, 8);
    _bytetrack_res_queue = new ProcessQueue<ByteTrackResult>(res_qsz);
    // Build per-stream track-rate map from the RTSP source list.
    // RtspUrlParams.rate now controls how often inference runs (<=1 = every frame).
    std::map<size_t, int> track_rate_map;
    for (const auto& p : rtsp_sources)
        track_rate_map[std::hash<std::string>{}(p.rtsp_url)] = p.rate;

    // Separate full-frame queue for the video encoder (not rate-filtered).
    _encode_raw_queue    = new ProcessQueue<ViDecSurfaceCudaBuff>(dec_qsz);
    _bytetrack_processor = new ByteTrackModelProcessor(
        cu_context,
        bytetrack_params.resize_h, bytetrack_params.resize_w,
        bytetrack_params.pre_maxBatch, bytetrack_params.pre_qsz,
        bytetrack_params.infer_maxBatch, bytetrack_params.custom_outputs,
        bytetrack_params.onnx_file, bytetrack_params.engine_file,
        _video_queue, _jpeg_queue, _encode_raw_queue,
        _bytetrack_res_queue, frame_rate, std::move(track_rate_map));
}

PipeLineProcessorByteTrack::~PipeLineProcessorByteTrack()
{
    _bytetrack_processor->StopModelThread();
    delete _bytetrack_processor;

    for (const auto& p : _video_processores) delete p.second;

    _p_thread_jpeg.set_status(true);
    _p_thread_jpeg.join();

    _p_thread_encode.set_status(true);
    _p_thread_encode.join();

    delete _jpeg_encoder;
    delete _video_queue;
    delete _jpeg_queue;
    delete _bytetrack_res_queue;
    delete _encode_raw_queue;
}

int PipeLineProcessorByteTrack::InitVideoStreams(const int dev_id)
{
    _video_queue = new ProcessQueue<ViDecSurfaceCudaBuff>(_dec_qsz);
    for (size_t i = 0; i < _rtsp_sources.size(); ++i) {
        const int ret = AddVideoStream(_video_processores, _video_queue, _cu_context, dev_id, _rtsp_sources[i], i);
        if (ret < 0)
            logger->warn("[{} {}]: Init stream {} failed: {}", __FUNCTION__, __LINE__, i, _rtsp_sources[i].rtsp_url);
        else
            _video_reset_count[i] = 0;
    }
    return 0;
}

int PipeLineProcessorByteTrack::JpegEncode()
{
    _p_thread_jpeg.set_status(false);
    while (!_p_thread_jpeg.get_status()) {
        const int n = waitEnoughBatch(_jpeg_queue, _p_jpeg_maxBatch, 20, _p_thread_jpeg.get_status());
        if (n <= 0) continue;
        auto frames = _jpeg_queue->get(n).get();
        std::vector<unsigned char> output;
        std::vector<size_t> lengths;
        _jpeg_encoder->EncodeSurfaceBatch(frames, output, lengths);
        _p_jpegEncodeFrames += n;
    }
    return 0;
}

int PipeLineProcessorByteTrack::VideoEncode()
{
    if (_encode_output_dir.empty()) return 0;
    FolderExist(_encode_output_dir + "/placeholder");

    // Push CUDA context for this thread: cudaMemcpy uses the driver-API context.
    CudaCtxPush ctxPush(_cu_context);

    // Per-stream VideoWriter (keyed by rtsp URL hash).
    std::map<size_t, cv::VideoWriter> writers;
    // Reusable host buffer for D2H download.
    std::vector<uint8_t> yuv_buf;

    _p_thread_encode.set_status(false);
    while (!_p_thread_encode.get_status())
    {
        // Read from the raw (full-rate) frame queue. This queue receives every
        // decoded frame regardless of track_rate, ensuring the encoded video is
        // always the complete original stream.
        //
        // Back-pressure protection: if the CPU encoder cannot keep up, drain the
        // queue to the latest frame to avoid ever-growing GPU-memory accumulation.
        // We keep at most 1 unconsumed entry; anything older is dropped silently.
        {
            const size_t depth = _encode_raw_queue->size();
            if (depth > 4) {
                // Queue is backing up — discard stale frames in bulk.
                auto stale = _encode_raw_queue->get(depth - 1).get();
                (void)stale;
            }
        }
        auto frame = _encode_raw_queue->get(std::chrono::milliseconds(100)).get();
        if (frame.Empty()) continue;

        const int    img_w     = static_cast<int>(frame.GetWidth());
        const int    img_h     = static_cast<int>(frame.GetHeight());
        const size_t yuv_bytes = frame.Total();

        yuv_buf.resize(yuv_bytes);
        cudaError_t err = cudaMemcpy(yuv_buf.data(),
                                     reinterpret_cast<const void*>(frame.GetGpuMem()),
                                     yuv_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            logger->error("[{} {}]: cudaMemcpy D2H failed: {}", __FUNCTION__, __LINE__,
                          cudaGetErrorString(err));
            continue;
        }

        // Choose correct YUV→BGR conversion based on pixel format.
        //   YUV420 (I420/IYUV): planar Y,U,V
        //   NV12:                semi-planar Y + interleaved UV
        cv::Mat yuv_mat(img_h * 3 / 2, img_w, CV_8UC1, yuv_buf.data());
        cv::Mat bgr;
        const Pixel_Format pix_fmt = frame.GetPixelFormat();
        if (pix_fmt == NV12)
            cv::cvtColor(yuv_mat, bgr, cv::COLOR_YUV2BGR_NV12);
        else
            cv::cvtColor(yuv_mat, bgr, cv::COLOR_YUV2BGR_IYUV);

        // Fetch the latest tracking result for this stream.
        // For non-tracked intermediate frames the previous result is reused,
        // so the viewer always sees bounding boxes even at reduced track_rate.
        const size_t hash = std::hash<std::string>{}(frame.GetRtspInfo());
        const StreamTrackState tracks = _bytetrack_processor->GetLatestTracks(hash);

        // Draw track bounding boxes.
        for (size_t i = 0; i < tracks.rects.size(); ++i)
        {
            const Rect& r  = tracks.rects[i];
            const int   id = tracks.trackIds[i];
            if (r.width <= 0 || r.height <= 0) continue;

            // Clamp rect to image boundary to avoid OpenCV assertion failures.
            const cv::Rect cv_rect(
                std::max(r.x, 0), std::max(r.y, 0),
                std::min(r.width,  img_w - std::max(r.x, 0)),
                std::min(r.height, img_h - std::max(r.y, 0)));
            if (cv_rect.width <= 0 || cv_rect.height <= 0) continue;

            cv::Scalar color = track_color(id);
            cv::rectangle(bgr, cv_rect, color, 2);

            std::string label = "ID:" + std::to_string(id);
            if (i < tracks.rectsInfo.size() && tracks.rectsInfo[i].score > 0.f)
                label += " " + std::to_string(static_cast<int>(tracks.rectsInfo[i].score * 100)) + "%";

            const int text_y = std::max(cv_rect.y - 6, 14);
            cv::putText(bgr, label, cv::Point(cv_rect.x, text_y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv::LINE_AA);
        }

        // Get or create VideoWriter for this RTSP stream.
        if (writers.find(hash) == writers.end())
        {
            const int fps = (_frame_rate > 0) ? _frame_rate : 30;

            // Bug fix 3: try multiple codecs in order of compatibility.
            // mp4v (MPEG-4) requires a specific OpenCV build; MJPG (.avi) is
            // universally available and guaranteed to produce a playable file.
            struct CodecCandidate { int fourcc; const char* ext; };
            const CodecCandidate candidates[] = {
                { cv::VideoWriter::fourcc('m','p','4','v'), ".mp4" },
                { cv::VideoWriter::fourcc('M','J','P','G'), ".avi" },
                { cv::VideoWriter::fourcc('X','V','I','D'), ".avi" },
            };

            bool opened = false;
            for (const auto& c : candidates)
            {
                const std::string out_path = _encode_output_dir + "/track_"
                                           + std::to_string(hash) + c.ext;
                writers[hash].open(out_path, c.fourcc, fps, cv::Size(img_w, img_h));
                if (writers[hash].isOpened())
                {
                    logger->info("[{} {}]: Opened video writer (codec {:c}{:c}{:c}{:c}): {}",
                                 __FUNCTION__, __LINE__,
                                 c.fourcc & 0xFF, (c.fourcc >> 8) & 0xFF,
                                 (c.fourcc >> 16) & 0xFF, (c.fourcc >> 24) & 0xFF,
                                 out_path);
                    opened = true;
                    break;
                }
                writers[hash].release();
            }
            if (!opened)
                logger->error("[{} {}]: All codecs failed for stream hash {}", __FUNCTION__, __LINE__, hash);
        }

        if (writers[hash].isOpened())
            writers[hash].write(bgr);

        ++_p_encodeFrames;
    }

    for (auto& kv : writers) kv.second.release();
    return 0;
}

int PipeLineProcessorByteTrack::StartPipeline()
{
    _p_thread_jpeg   = TaskThread(std::thread([this]() { this->JpegEncode(); }));
    if (!_encode_output_dir.empty())
        _p_thread_encode = TaskThread(std::thread([this]() { this->VideoEncode(); }));
    _bytetrack_processor->StartModelThread();
    return 0;
}

std::vector<DecoderProcessorStats> PipeLineProcessorByteTrack::CollectVideoProcessorStats()
{
    std::vector<DecoderProcessorStats> stats;
    for (const auto& e : _video_processores) {
        DecoderProcessorStats stat;
        stat.id = e.first;
        if (e.second) {
            stat.mapFramesOK     = e.second->GetDecoderMapFramesOK();
            stat.receiveFramesOK = e.second->GetDecoderReceiveFramesOK();
        } else {
            stat.mapFramesOK = stat.receiveFramesOK = -1;
        }
        stats.push_back(stat);
    }
    return stats;
}

std::map<int, int> PipeLineProcessorByteTrack::ChcekDecoderStautsAndRestart()
{
    for (const auto& vp : _video_processores) {
        const int id = vp.first;
        if (vp.second->GetDecStatus() != 0) {
            RestartVideoStream(_video_processores, _video_queue, _cu_context, _dev_id, _rtsp_sources[id], id);
            _video_reset_count[id]++;
        }
    }
    return _video_reset_count;
}
