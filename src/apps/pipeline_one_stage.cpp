#include "pipeline_one_stage.h"

using namespace std;
static int AddVideoStream(std::map<int, VideoStreamProcessor*>& VideoStreamProcessorLists,
                          ProcessQueue<ViDecSurfaceCudaBuff>*   VideoStreamQueue,
                          CUcontext                             cuContext,
                          const int                             dev,
                          const RtspUrlParams&                  rtspParams,
                          int                                   id)  // frames per seconds
{
    size_t times = 0;
    while (times <= 100)
    {
        try
        {
            VideoStreamProcessorLists[id] =
                new VideoStreamProcessor(cuContext, dev, rtspParams.rtsp_url, VideoStreamQueue, id, rtspParams.rate);
            break;
        }
        catch (const std::exception& e)
        {
            delete VideoStreamProcessorLists[id];
            VideoStreamProcessorLists[id] = nullptr;
            times++;
            std::cerr << e.what() << "retry times " << times << '\n';
        }
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }
    if (times > 100)
        return -1;
    return 0;
}

static int RestartVideoStream(std::map<int, VideoStreamProcessor*>& VideoStreamProcessorLists,
                              ProcessQueue<ViDecSurfaceCudaBuff>*   VideoStreamQueue,
                              CUcontext                             cuContext,
                              const int                             dev,
                              const RtspUrlParams&                  rtspParams,
                              int                                   id)
{
    if (VideoStreamProcessorLists[id])
    {
        delete VideoStreamProcessorLists[id];
        VideoStreamProcessorLists[id] = nullptr;
    }

    int ret = AddVideoStream(VideoStreamProcessorLists, VideoStreamQueue, cuContext, dev, rtspParams, id);
    if (ret < 0)
    {
        logger->warn("[{} {}]: Restart id {} Rtsp Fail : %s", __FUNCTION__, __LINE__, id, rtspParams.rtsp_url);
    }
    return ret;
}

template <class T>
static T clip(T x, T min, T max)
{
    if (x > max)
        return max;
    if (x < min)
        return min;
    return x;
}

YoloV5ModelProcessor::YoloV5ModelProcessor(CUcontext                           cu_context,
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
                                           ProcessQueue<YoloV5Result>*         res_queue)
    : ModelProcessor(cu_context,
                     resize_h,
                     resize_w,
                     pre_maxBatch,
                     pre_qsz,
                     infer_maxBatch,
                     custom_outputs,
                     onnx_file,
                     engine_file)
    , _video_queue(video_queue)
    , _res_queue(res_queue)
    , _jpeg_queue(jpeg_queue)
{
    InitModelHandle();
}

YoloV5ModelProcessor::~YoloV5ModelProcessor()
{
    // StopModelThread();

    delete _m_preprocessor;
    _m_preprocessor = nullptr;

    delete _m_infer_model;
    _m_infer_model = nullptr;

    delete _m_preprocess_queue;
    _m_preprocess_queue = nullptr;
}

int YoloV5ModelProcessor::InitPreProcess()
{
    _m_PreproFrames.store(0);

    _m_preprocess_queue = new ProcessQueue<YoloV5PreSurfaces>(_m_pre_qsz);
    _m_preprocessor     = new DetectPreprocessor(_m_resize_h, _m_resize_w, _m_cu_context, _m_pre_maxBatch);
    return 0;
}

int YoloV5ModelProcessor::InitIxrtIfner()
{
    _m_InferFrames.store(0);

    std::string plugin_lib_path = "/usr/local/corex/lib/libixrt_plugin.so";
    _m_infer_model              = new Trt(plugin_lib_path);

    int opt_batch = _m_infer_maxBatch / 2;
    if (opt_batch % 2 != 0 || opt_batch <= 1)
    {
        opt_batch = std::max(2, opt_batch - (opt_batch % 2));
    }

    _m_infer_model->AddDynamicShapeProfile("images",
                                           {1, 3, _m_resize_h, _m_resize_w},
                                           {opt_batch, 3, _m_resize_h, _m_resize_w},
                                           {static_cast<int>(_m_infer_maxBatch), 3, _m_resize_h, _m_resize_w});
    _m_infer_model->CreateEngine(_m_onnx_file, _m_engine_file, _m_model_outputs, 1);

    _m_model_height = _m_infer_model->GetBindingDimsVec(0)[2];
    _m_model_width  = _m_infer_model->GetBindingDimsVec(0)[3];
    return 0;
}

void YoloV5ModelProcessor::CudaPreProcess()
{
    CudaCtxPush ctxPush(_m_cu_context);
    this->_m_thread_preprocess.set_status(false);
    while (!this->_m_thread_preprocess.get_status())
    {
        int get_num = waitEnoughBatch(_video_queue, _m_pre_maxBatch, 20, this->_m_thread_preprocess.get_status());
        if (get_num <= 0)
        {
            logger->warn("[{} {}]: no video frames to YoloV5 preprocess in 2S ", __FUNCTION__, __LINE__);
            continue;
        }

        auto oriSurfaces_future = _video_queue->get(get_num);
        auto oriSurfaces        = oriSurfaces_future.get();
        for (size_t i = 0; i < oriSurfaces.size(); i++)
        {
            auto vframe = oriSurfaces[i];
            _jpeg_queue->put(vframe);
        }

        SurfaceCudaBuff outGpuSurf = _m_preprocessor->Process(oriSurfaces);
        if ((outGpuSurf.Empty()) || (outGpuSurf.GetWidth() == 0) || (outGpuSurf.GetHeight() == 0))
        {
            logger->warn("[{} {}]: YoloV5 preprocess frames is empty !!!", __FUNCTION__, __LINE__);
            continue;
        }

        YoloV5PreSurfaces toInferSurface = {outGpuSurf, oriSurfaces};
        _m_preprocess_queue->put(toInferSurface);
        _m_PreproFrames += get_num;
    }
    return;
}

void YoloV5ModelProcessor::IxrtInfer()
{
    int         number_from_q = 1;
    CudaCtxPush ctxPush(_m_cu_context);
    this->_m_thread_infer.set_status(false);

    while (!this->_m_thread_infer.get_status())
    {
        int get_num = waitEnoughBatch(_m_preprocess_queue, number_from_q, 20, this->_m_thread_infer.get_status());
        if (get_num <= 0)
        {
            logger->warn("[{} {}]:no YoloV5 preprocess frames to YoloV5 infer in 2S !!!", __FUNCTION__, __LINE__);
            continue;
        }

        auto            preSurf_future = _m_preprocess_queue->get();
        auto            preSurf        = preSurf_future.get();
        SurfaceCudaBuff preframe       = preSurf.preSurface;
        int             batch          = preframe.GetBatch();

        if (preframe.Empty())
        {
            ostringstream errorString;
            errorString << endl << __FUNCTION__ << " " << __LINE__ << ": YoloV5 PreProcess Frames is Empty !!!" << endl;
            throw runtime_error(errorString.str());
        }

        std::vector<void*> buffers;

        buffers.resize(_m_infer_model->nbBindings);
        size_t input0_memSize  = _m_infer_model->mBindingSize[0] * batch;
        size_t output0_memSize = _m_infer_model->mBindingSize[1] * batch;
        size_t output1_memSize = _m_infer_model->mBindingSize[2] * batch;

        std::vector<float>   output0(output0_memSize / sizeof(float));
        std::vector<int32_t> output1(output1_memSize / sizeof(int32_t));

        checkCudaErrors(cudaMemcpyAsync(_m_infer_model->GetBindingPtr(0),
                                        reinterpret_cast<void*>(preframe.GetGpuMem()),
                                        input0_memSize,
                                        cudaMemcpyDeviceToDevice,
                                        this->_m_infer_cudastream));
        buffers[0] = _m_infer_model->GetBindingPtr(0);
        // buffers[0] = reinterpret_cast<void*>(preframe.GetGpuMem());

        buffers[1] = _m_infer_model->GetBindingPtr(1);
        buffers[2] = _m_infer_model->GetBindingPtr(2);

        _m_infer_model->Forward(&buffers[0], batch, this->_m_infer_cudastream);
        checkCudaErrors(cudaMemcpyAsync(
            output0.data(), buffers[1], output0_memSize, cudaMemcpyDeviceToHost, this->_m_infer_cudastream));
        checkCudaErrors(cudaMemcpyAsync(
            output1.data(), buffers[2], output1_memSize, cudaMemcpyDeviceToHost, this->_m_infer_cudastream));
        checkCudaErrors(cudaStreamSynchronize(this->_m_infer_cudastream));
        _m_InferFrames += batch;

        if (preSurf.oriSurfaces.size() != batch)
        {
            std::ostringstream errorString;
            errorString << __FUNCTION__ << __LINE__ << std::endl
                        << "original images size(" << preSurf.oriSurfaces.size() << ") != YoloV5 pre size(" << batch
                        << ")" << std::endl;
            throw std::runtime_error(errorString.str());
        }

        int box_num = _m_infer_model->GetBindingDimsVec(1)[1];
        for (size_t i = 0; i < batch; i++)
        {
            std::vector<RectInfo> rectsInfo;
            std::vector<Rect>     rects;

            ViDecSurfaceCudaBuff oriSurface     = preSurf.oriSurfaces[i];
            int                  oriSurf_width  = oriSurface.GetWidth();
            int                  oriSurf_height = oriSurface.GetHeight();

            double ratioW = static_cast<double>(_m_model_width) / static_cast<double>(oriSurf_width);
            double ratioH = static_cast<double>(_m_model_height) / static_cast<double>(oriSurf_height);

            double ratio = std::min(ratioW, ratioH);

            for (size_t j = 0; j < output1[i]; j++)
            {
                // printf("batch：%d, box label:%f, score:%f  box:%f %f %f %f\n",
                //        i,
                //        output0[i * 6 * box_num + j * 6 + 4],
                //        output0[i * 6 * box_num + j * 6 + 5],
                //        output0[i * 6 * box_num + j * 6 + 0],
                //        output0[i * 6 * box_num + j * 6 + 1],
                //        output0[i * 6 * box_num + j * 6 + 2],
                //        output0[i * 6 * box_num + j * 6 + 3]);

                int box_left =
                    clip<int>(static_cast<int>(output0[i * 6 * box_num + j * 6 + 0] / ratio), 0, oriSurf_width);
                int box_top =
                    clip<int>(static_cast<int>(output0[i * 6 * box_num + j * 6 + 1] / ratio), 0, oriSurf_height);
                int box_right =
                    clip<int>(static_cast<int>(output0[i * 6 * box_num + j * 6 + 2] / ratio), 0, oriSurf_width);
                int box_bottom =
                    clip<int>(static_cast<int>(output0[i * 6 * box_num + j * 6 + 3] / ratio), 0, oriSurf_height);

                Rect rect;
                rect.x      = box_left;
                rect.y      = box_top;
                rect.width  = box_right - box_left;
                rect.height = box_bottom - box_top;
                if (rect.width <= 5 || rect.height <= 5 || rect.width > oriSurf_width || rect.height > oriSurf_height)
                    continue;
                rects.push_back(rect);

                int      label  = static_cast<int>(output0[i * 6 * box_num + j * 6 + 4]);
                float    score  = output0[i * 6 * box_num + j * 6 + 5];
                RectInfo result = {label, score};
                rectsInfo.push_back(result);

                if (rectsInfo.size() >= 100)
                {
                    logger->warn("[{} {}]: YoloV5 detect rects {} > 100 !!!", __FUNCTION__, __LINE__, output0[i]);
                    break;
                }
            }

            if (rects.size() > 0)
            {
                YoloV5Result res_ = {oriSurface, rects, rectsInfo};
                _res_queue->put(res_);
            }
        }
    }
    return;
}

PipeLineProcessorOneStage::PipeLineProcessorOneStage(CUcontext                         cu_context,
                                                     const int                         dev_id,
                                                     const int                         jpeg_maxBatch,
                                                     const int                         jpeg_qsz,
                                                     const int                         dec_qsz,
                                                     const std::vector<RtspUrlParams>& rtsp_sources,
                                                     const ModelParams&                yolov5_params)
    : _cu_context(cu_context)
    , _rtsp_sources(rtsp_sources)
    , _dec_qsz(dec_qsz)
    , _dev_id(dev_id)
    , _p_jpeg_maxBatch(jpeg_maxBatch)
    , _jpeg_qsz(jpeg_qsz)
{
    CudaCtxPush ctxPush(_cu_context);
    InitVideoStreams(dev_id);

    _jpeg_queue   = new ProcessQueue<ViDecSurfaceCudaBuff>(jpeg_qsz);
    _jpeg_encoder = new IluvatarJpegCodec(dev_id, jpeg_maxBatch, false, true);

    _yolov5_res_queue = new ProcessQueue<YoloV5Result>(dec_qsz);
    _yolov5_processor = new YoloV5ModelProcessor(cu_context,
                                                 yolov5_params.resize_h,
                                                 yolov5_params.resize_w,
                                                 yolov5_params.pre_maxBatch,
                                                 yolov5_params.pre_qsz,
                                                 yolov5_params.infer_maxBatch,
                                                 yolov5_params.custom_outputs,
                                                 yolov5_params.onnx_file,
                                                 yolov5_params.engine_file,
                                                 _video_queue,
                                                 _jpeg_queue,
                                                 _yolov5_res_queue);
}

PipeLineProcessorOneStage::~PipeLineProcessorOneStage()
{
    _yolov5_processor->StopModelThread();
    delete _yolov5_processor;

    for (const auto& processor : _video_processores)
    {
        delete processor.second;
    }

    this->_p_thread_jpeg.set_status(true);
    this->_p_thread_jpeg.join();

    delete _jpeg_encoder;
    _jpeg_encoder = nullptr;

    delete _video_queue;
    _video_queue = nullptr;

    delete _jpeg_queue;
    _jpeg_queue = nullptr;

    delete _yolov5_res_queue;
    _yolov5_res_queue = nullptr;
}

int PipeLineProcessorOneStage::InitVideoStreams(const int dev_id)
{
    _video_queue = new ProcessQueue<ViDecSurfaceCudaBuff>(_dec_qsz);
    for (size_t i = 0; i < _rtsp_sources.size(); i++)
    {
        int ret = AddVideoStream(_video_processores, _video_queue, _cu_context, dev_id, _rtsp_sources[i], i);
        if (ret < 0)
        {
            // need to handle
            logger->warn("[{} {}]: Init id {} Rtsp Fail : {}", __FUNCTION__, __LINE__, i, _rtsp_sources[i].rtsp_url);
            continue;
        }
        _video_reset_count[i] = 0;
    }
    return 0;
}

int PipeLineProcessorOneStage::JpegEncode()
{
    this->_p_thread_jpeg.set_status(false);

    while (!this->_p_thread_jpeg.get_status())
    {
        int get_num = waitEnoughBatch(_jpeg_queue, _p_jpeg_maxBatch, 20, this->_p_thread_jpeg.get_status());
        if (get_num <= 0)
        {
            logger->warn("[{} {}]: no video frames to jpeg encode in 2S", __FUNCTION__, __LINE__);
            continue;
        }

        auto preframe_future = _jpeg_queue->get(get_num);
        auto vframe_vec      = preframe_future.get();

        std::vector<unsigned char> output;
        std::vector<size_t>        lengthes;
        int                        ret = _jpeg_encoder->EncodeSurfaceBatch(vframe_vec, output, lengthes);

        // std::string                out_name = "./enc" + std::to_string(_p_jpegEncodeFrames.load()) + ".jpeg";
        // if (_jpeg_encoder->SaveRawData(out_name, jpeg_mat) < 0)
        // {
        //     std::ostringstream errorString;
        //     errorString << __FUNCTION__ << __LINE__ << std::endl << "jpeg encoder save error !!!" << std::endl;
        //     throw std::runtime_error(errorString.str());
        // }
        _p_jpegEncodeFrames += get_num;
    }
    return 0;
}

int PipeLineProcessorOneStage::StartPipeline()
{
    // _video_queue->reset_count();
    this->_p_thread_jpeg = TaskThread(std::thread([this]() { this->JpegEncode(); }));
    _yolov5_processor->StartModelThread();

    return 0;
}

std::vector<DecoderProcessorStats> PipeLineProcessorOneStage::CollectVideoProcessorStats()
{
    std::vector<DecoderProcessorStats> stats;

    for (const auto& entry : _video_processores)
    {
        int                   id        = entry.first;
        VideoStreamProcessor* processor = entry.second;
        DecoderProcessorStats stat;
        stat.id = id;
        if (processor != nullptr)
        {
            stat.mapFramesOK     = processor->GetDecoderMapFramesOK();
            stat.receiveFramesOK = processor->GetDecoderReceiveFramesOK();
        }
        else
        {
            stat.mapFramesOK     = -1;
            stat.receiveFramesOK = -1;
        }
        stats.push_back(stat);
    }

    return stats;
}

std::map<int, int> PipeLineProcessorOneStage::ChcekDecoderStautsAndRestart()
{
    for (const auto& video_processor : _video_processores)
    {
        int                   id        = video_processor.first;
        VideoStreamProcessor* processor = video_processor.second;
        if (processor->GetDecStatus() != 0)
        {
            int ret = RestartVideoStream(_video_processores, _video_queue, _cu_context, _dev_id, _rtsp_sources[id], id);
            if (ret < 0)
            {
                logger->warn("[{} {}]: rtsp id:{} restart fail", __FUNCTION__, __LINE__, id);
            }
            _video_reset_count[id] += 1;
        }
    }

    return _video_reset_count;
}