#include "pipeline_two_stage.h"

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

PPYoloEModelProcessor::PPYoloEModelProcessor(CUcontext                           cu_context,
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
                                             ProcessQueue<PPYoloEResult>*        res_queue)
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

PPYoloEModelProcessor::~PPYoloEModelProcessor()
{
    // StopModelThread();

    delete _m_preprocessor;
    _m_preprocessor = nullptr;

    delete _m_infer_model;
    _m_infer_model = nullptr;

    delete _m_preprocess_queue;
    _m_preprocess_queue = nullptr;
}

int PPYoloEModelProcessor::InitPreProcess()
{
    _m_PreproFrames.store(0);

    _m_preprocess_queue = new ProcessQueue<PPYoloEPreSurfaces>(_m_pre_qsz);
    _m_preprocessor     = new DetectPreprocessor(_m_resize_h, _m_resize_w, _m_cu_context, _m_pre_maxBatch);
    return 0;
}

int PPYoloEModelProcessor::InitIxrtIfner()
{
    _m_InferFrames.store(0);

    std::string plugin_lib_path = "/usr/local/corex/lib/liboss_ixrt_plugin.so";
    _m_infer_model              = new Trt(plugin_lib_path);

    int opt_batch = _m_infer_maxBatch / 2;
    if (opt_batch % 2 != 0 || opt_batch <= 1)
    {
        opt_batch = std::max(2, opt_batch - (opt_batch % 2));
    }

    _m_infer_model->AddDynamicShapeProfile("image",
                                           {1, 3, _m_resize_h, _m_resize_w},
                                           {opt_batch, 3, _m_resize_h, _m_resize_w},
                                           {static_cast<int>(_m_infer_maxBatch), 3, _m_resize_h, _m_resize_w});
    _m_infer_model->CreateEngine(_m_onnx_file, _m_engine_file, _m_model_outputs, 1);

    _m_model_height = _m_infer_model->GetBindingDimsVec(0)[2];
    _m_model_width  = _m_infer_model->GetBindingDimsVec(0)[3];
    return 0;
}

void PPYoloEModelProcessor::CudaPreProcess()
{
    CudaCtxPush ctxPush(_m_cu_context);
    this->_m_thread_preprocess.set_status(false);
    while (!this->_m_thread_preprocess.get_status())
    {
        int get_num = waitEnoughBatch(_video_queue, _m_pre_maxBatch, 20, this->_m_thread_preprocess.get_status());
        if (get_num <= 0)
        {
            logger->warn("[{} {}]: no video frames to ppyoloe preprocess in 2S ", __FUNCTION__, __LINE__);
            continue;
        }

        size_t width      = 0;
        size_t height     = 0;
        bool   diff_shape = false;

        auto oriSurfaces_future = _video_queue->get(get_num);
        auto oriSurfaces        = oriSurfaces_future.get();
        for (size_t i = 0; i < oriSurfaces.size(); i++)
        {
            auto vframe = oriSurfaces[i];
            width       = vframe.GetWidth();
            height      = vframe.GetHeight();
            if (vframe.GetWidth() != width || vframe.GetHeight() != height)
            {
                diff_shape = true;
            }
            _jpeg_queue->put(vframe);
        }

        SurfaceCudaBuff outGpuSurf = _m_preprocessor->Process(oriSurfaces);
        if ((outGpuSurf.Empty()) || (outGpuSurf.GetWidth() == 0) || (outGpuSurf.GetHeight() == 0))
        {
            logger->warn("[{} {}]: ppyoloe preprocess frames is empty !!!", __FUNCTION__, __LINE__);
            continue;
        }

        PPYoloEPreSurfaces toInferSurface = {outGpuSurf, oriSurfaces};
        _m_preprocess_queue->put(toInferSurface);
        _m_PreproFrames += get_num;
    }
    return;
}

void PPYoloEModelProcessor::IxrtInfer()
{
    int         number_from_q = 1;
    CudaCtxPush ctxPush(_m_cu_context);
    this->_m_thread_infer.set_status(false);

    while (!this->_m_thread_infer.get_status())
    {
        int get_num = waitEnoughBatch(_m_preprocess_queue, number_from_q, 20, this->_m_thread_infer.get_status());
        if (get_num <= 0)
        {
            logger->warn("[{} {}]:no ppyoloe preprocess frames to ppyoloe infer in 2S !!!", __FUNCTION__, __LINE__);
            continue;
        }

        auto            preSurf_future = _m_preprocess_queue->get();
        auto            preSurf        = preSurf_future.get();
        SurfaceCudaBuff preframe       = preSurf.preSurface;
        int             batch          = preframe.GetBatch();

        if (preframe.Empty())
        {
            ostringstream errorString;
            errorString << endl
                        << __FUNCTION__ << " " << __LINE__ << ": PPYoloE PreProcess Frames is Empty !!!" << endl;
            throw runtime_error(errorString.str());
        }

        std::vector<float> output0, output1, output2, output3;
        std::vector<void*> buffers;

        buffers.resize(_m_infer_model->nbBindings);
        size_t input0_memSize  = _m_infer_model->mBindingSize[0] * batch;
        size_t output0_memSize = _m_infer_model->mBindingSize[1] * batch;
        size_t output1_memSize = _m_infer_model->mBindingSize[2] * batch;
        size_t output2_memSize = _m_infer_model->mBindingSize[3] * batch;
        size_t output3_memSize = _m_infer_model->mBindingSize[4] * batch;

        output0.resize(output0_memSize / sizeof(float));
        output1.resize(output1_memSize / sizeof(float));
        output2.resize(output2_memSize / sizeof(float));
        output3.resize(output3_memSize / sizeof(float));

        checkCudaErrors(cudaMemcpyAsync(_m_infer_model->GetBindingPtr(0),
                                        reinterpret_cast<void*>(preframe.GetGpuMem()),
                                        input0_memSize,
                                        cudaMemcpyDeviceToDevice,
                                        this->_m_infer_cudastream));
        buffers[0] = _m_infer_model->GetBindingPtr(0);
        // buffers[0] = reinterpret_cast<void*>(preframe.GetGpuMem());

        buffers[1] = _m_infer_model->GetBindingPtr(1);
        buffers[2] = _m_infer_model->GetBindingPtr(2);
        buffers[3] = _m_infer_model->GetBindingPtr(3);
        buffers[4] = _m_infer_model->GetBindingPtr(4);

        _m_infer_model->Forward(&buffers[0], batch, this->_m_infer_cudastream);
        checkCudaErrors(cudaMemcpyAsync(
            output0.data(), buffers[1], output0_memSize, cudaMemcpyDeviceToHost, this->_m_infer_cudastream));
        checkCudaErrors(cudaMemcpyAsync(
            output1.data(), buffers[2], output1_memSize, cudaMemcpyDeviceToHost, this->_m_infer_cudastream));
        checkCudaErrors(cudaMemcpyAsync(
            output2.data(), buffers[3], output2_memSize, cudaMemcpyDeviceToHost, this->_m_infer_cudastream));
        checkCudaErrors(cudaMemcpyAsync(
            output3.data(), buffers[4], output3_memSize, cudaMemcpyDeviceToHost, this->_m_infer_cudastream));
        checkCudaErrors(cudaStreamSynchronize(this->_m_infer_cudastream));
        _m_InferFrames += batch;

        if (preSurf.oriSurfaces.size() != batch)
        {
            std::ostringstream errorString;
            errorString << __FUNCTION__ << __LINE__ << std::endl
                        << "original images size(" << preSurf.oriSurfaces.size() << ") != ppyoloe pre size(" << batch
                        << ")" << std::endl;
            throw std::runtime_error(errorString.str());
        }

        for (size_t i = 0; i < batch; i++)
        {
            std::vector<RectInfo> rectsInfo;
            std::vector<Rect>     rects;

            ViDecSurfaceCudaBuff oriSurface     = preSurf.oriSurfaces[i];
            int                  oriSurf_width  = oriSurface.GetWidth();
            int                  oriSurf_height = oriSurface.GetHeight();

            double ratioW = static_cast<double>(_m_model_width) / oriSurf_width;
            double ratioH = static_cast<double>(_m_model_height) / oriSurf_height;

            double ratio = std::min(ratioW, ratioH);

            for (size_t j = 0; j < output0[i]; j++)
            {
                // printf("box label:%f, score:%f  box:%f %f %f %f\n",
                //        output3[i * 1000 + j],
                //        output2[i * 1000 + j],
                //        output1[i * 4000 + j * 4 + 0],
                //        output1[i * 4000 + j * 4 + 1],
                //        output1[i * 4000 + j * 4 + 2],
                //        output1[i * 4000 + j * 4 + 3]);

                int box_left   = clip<int>(static_cast<int>(output1[i * 4000 + j * 4 + 0] / ratio), 0, oriSurf_width);
                int box_top    = clip<int>(static_cast<int>(output1[i * 4000 + j * 4 + 1] / ratio), 0, oriSurf_height);
                int box_right  = clip<int>(static_cast<int>(output1[i * 4000 + j * 4 + 2] / ratio), 0, oriSurf_width);
                int box_bottom = clip<int>(static_cast<int>(output1[i * 4000 + j * 4 + 3] / ratio), 0, oriSurf_height);

                Rect rect;
                rect.x      = box_left;
                rect.y      = box_top;
                rect.width  = box_right - box_left;
                rect.height = box_bottom - box_top;
                if (rect.width <= 5 || rect.height <= 5 || rect.width > oriSurf_width || rect.height > oriSurf_height)
                    continue;
                rects.push_back(rect);

                int      label  = static_cast<int>(output3[i * 1000 + j]);
                float    score  = output2[i * 1000 + j];
                RectInfo result = {label, score};
                rectsInfo.push_back(result);

                if (rectsInfo.size() >= 100)
                {
                    logger->warn("[{} {}]: ppyoloe detect rects {} > 100 !!!", __FUNCTION__, __LINE__, output0[i]);
                    break;
                }
            }

            if (rects.size() > 0)
            {
                PPYoloEResult res_ = {oriSurface, rects, rectsInfo};
                _res_queue->put(res_);
            }
        }
    }
    return;
}

PPLCNetModelProcessor::PPLCNetModelProcessor(CUcontext                       cu_context,
                                             int                             resize_h,
                                             int                             resize_w,
                                             int                             pre_maxBatch,
                                             int                             pre_qsz,
                                             int                             infer_maxBatch,
                                             const std::vector<std::string>& custom_outputs,
                                             const std::string&              onnx_file,
                                             const std::string&              engine_file,
                                             ProcessQueue<PPYoloEResult>*    input_queue)
    : ModelProcessor(cu_context,
                     resize_h,
                     resize_w,
                     pre_maxBatch,
                     pre_qsz,
                     infer_maxBatch,
                     custom_outputs,
                     onnx_file,
                     engine_file)
    , _input_queue(input_queue)
{
    InitModelHandle();
}

PPLCNetModelProcessor::~PPLCNetModelProcessor()
{
    // StopModelThread();
    delete _m_preprocessor;
    _m_preprocessor = nullptr;

    delete _m_infer_model;
    _m_infer_model = nullptr;

    delete _m_preprocess_queue;
    _m_preprocess_queue = nullptr;
}

int PPLCNetModelProcessor::InitPreProcess()
{
    _m_PreproFrames.store(0);

    _m_preprocess_queue = new ProcessQueue<SurfaceCudaBuff>(_m_pre_qsz);
    _m_preprocessor     = new ClassifyPreprocessor(_m_resize_h, _m_resize_w, _m_cu_context, _m_pre_maxBatch);
    return 0;
}

int PPLCNetModelProcessor::InitIxrtIfner()
{
    _m_InferFrames.store(0);

    _m_infer_model = new Trt();

    int opt_batch = _m_infer_maxBatch / 2;
    if (opt_batch % 2 != 0 || opt_batch <= 1)
    {
        opt_batch = std::max(2, opt_batch - (opt_batch % 2));
    }

    _m_infer_model->AddDynamicShapeProfile(
        "x", {1, 3, _m_resize_h, _m_resize_w}, {opt_batch, 3, _m_resize_h, _m_resize_w}, {static_cast<int>(_m_infer_maxBatch), 3, _m_resize_h, _m_resize_w});
    _m_infer_model->CreateEngine(_m_onnx_file, _m_engine_file, _m_model_outputs, 1);

    _m_model_height = _m_infer_model->GetBindingDimsVec(0)[2];
    _m_model_width  = _m_infer_model->GetBindingDimsVec(0)[3];

    return 0;
}

void PPLCNetModelProcessor::CudaPreProcess()
{
    CudaCtxPush ctxPush(_m_cu_context);
    this->_m_thread_preprocess.set_status(false);

    while (!this->_m_thread_preprocess.get_status())
    {
        int get_num = waitEnoughBatch(_input_queue, _m_pre_maxBatch, 20, this->_m_thread_preprocess.get_status());
        if (get_num <= 0)
        {
            logger->warn("[{} {}]: no ppyoloe out frames to pplcnet preprocess in 2S", __FUNCTION__, __LINE__);
            continue;
        }
        std::vector<ViDecSurfaceCudaBuff> oriSurfaces_vec;
        std::vector<std::vector<Rect>>    rects_vec;

        auto yoloe_results_future = _input_queue->get(get_num);
        auto yoloe_results        = yoloe_results_future.get();

        for (size_t i = 0; i < yoloe_results.size(); i++)
        {
            auto                 yoloe_result = yoloe_results[i];
            ViDecSurfaceCudaBuff oriSurfaces  = yoloe_result.oriSurfaces;
            if (oriSurfaces.Empty())
            {
                ostringstream errorString;
                errorString << endl
                            << __FUNCTION__ << " " << __LINE__ << ": Videc input [" << i << "]: is Empty !!!" << endl;
                throw runtime_error(errorString.str());
            }

            std::vector<Rect> rects = yoloe_result.rects;
            if (rects.size() <= 0)
                continue;
            oriSurfaces_vec.push_back(oriSurfaces);
            rects_vec.push_back(rects);
        }

        if (oriSurfaces_vec.size() > 0)
        {
            SurfaceCudaBuff outGpuSurfs = _m_preprocessor->Process(oriSurfaces_vec, rects_vec);

            if ((outGpuSurfs.Empty()) || (outGpuSurfs.GetWidth() == 0) && (outGpuSurfs.GetHeight() == 0))
            {
                logger->warn("[{} {}]: pplcnet preprocess frames is empty !!!", __FUNCTION__, __LINE__);
                continue;
            }
            _m_preprocess_queue->put(outGpuSurfs);
        }

        _m_PreproFrames += get_num;
    }
    return;
}

void processBatch(void*        preframe_gpu_mem,
                  size_t       batch_size,
                  size_t       batch_offset,
                  Trt*         infer_model,
                  cudaStream_t _cudastream)
{
    std::vector<float> output0;
    std::vector<void*> buffers;

    buffers.resize(infer_model->nbBindings);
    size_t input0_memSize  = infer_model->mBindingSize[0] * batch_size;
    size_t output0_memSize = infer_model->mBindingSize[1] * batch_size;

    checkCudaErrors(
        cudaMemcpyAsync(infer_model->GetBindingPtr(0),
                        reinterpret_cast<void*>(preframe_gpu_mem + batch_offset * infer_model->mBindingSize[0]),
                        input0_memSize,
                        cudaMemcpyDeviceToDevice,
                        _cudastream));

    // buffers[0] = reinterpret_cast<void*>(preframe_gpu_mem + batch_offset * binding_size);
    buffers[0] = infer_model->GetBindingPtr(0);
    buffers[1] = infer_model->GetBindingPtr(1);

    infer_model->Forward(&buffers[0], batch_size, _cudastream);

    output0.resize(output0_memSize / sizeof(float));
    checkCudaErrors(cudaMemcpyAsync(output0.data(), buffers[1], output0_memSize, cudaMemcpyDeviceToHost, _cudastream));
    checkCudaErrors(cudaStreamSynchronize(_cudastream));

    return;
}

void PPLCNetModelProcessor::IxrtInfer()
{
    CudaCtxPush ctxPush(_m_cu_context);
    this->_m_thread_infer.set_status(false);

    while (!this->_m_thread_infer.get_status())
    {
        int get_num = waitEnoughBatch(_m_preprocess_queue, 1, 40, this->_m_thread_infer.get_status());
        if (get_num <= 0)
        {
            logger->warn("[{} {}]: no pplcnet preprocess frames to pplcnet infer in 4S", __FUNCTION__, __LINE__);
            continue;
        }

        auto preframe_future = _m_preprocess_queue->get();
        auto preframe        = preframe_future.get();

        if (preframe.Empty())
        {
            ostringstream errorString;
            errorString << endl
                        << __FUNCTION__ << " " << __LINE__ << ": PPLCNet PreProcess Frames is Empty !!!" << endl;
            throw runtime_error(errorString.str());
        }

        int batch = preframe.GetBatch();
        if (batch > _m_infer_maxBatch)
        {
            size_t numFullBatches = batch / _m_infer_maxBatch;
            size_t remainingBatch = batch % _m_infer_maxBatch;

            for (size_t i = 0; i < numFullBatches; ++i)
            {
                processBatch(reinterpret_cast<void*>(preframe.GetGpuMem()),
                             _m_infer_maxBatch,
                             i * _m_infer_maxBatch,
                             _m_infer_model,
                             this->_m_infer_cudastream);
            }

            if (remainingBatch > 0)
            {
                processBatch(reinterpret_cast<void*>(preframe.GetGpuMem()),
                             remainingBatch,
                             numFullBatches * _m_infer_maxBatch,
                             _m_infer_model,
                             this->_m_infer_cudastream);
            }
        }
        else
        {
            processBatch(
                reinterpret_cast<void*>(preframe.GetGpuMem()), batch, 0, _m_infer_model, this->_m_infer_cudastream);
        }

        _m_InferFrames += batch;
    }
    return;
}

PipeLineProcessorTwoStage::PipeLineProcessorTwoStage(CUcontext                         cu_context,
                                                     const int                         dev_id,
                                                     const int                         jpeg_maxBatch,
                                                     const int                         jpeg_qsz,
                                                     const int                         dec_qsz,
                                                     const std::vector<RtspUrlParams>& rtsp_sources,
                                                     const ModelParams&                ppyoloe_params,
                                                     const ModelParams&                pplcnet_params)
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

    _ppyoloe_res_queue = new ProcessQueue<PPYoloEResult>(dec_qsz);
    _ppyoloe_processor = new PPYoloEModelProcessor(cu_context,
                                                   ppyoloe_params.resize_h,
                                                   ppyoloe_params.resize_w,
                                                   ppyoloe_params.pre_maxBatch,
                                                   ppyoloe_params.pre_qsz,
                                                   ppyoloe_params.infer_maxBatch,
                                                   ppyoloe_params.custom_outputs,
                                                   ppyoloe_params.onnx_file,
                                                   ppyoloe_params.engine_file,
                                                   _video_queue,
                                                   _jpeg_queue,
                                                   _ppyoloe_res_queue);

    _pplcnet_processor = new PPLCNetModelProcessor(cu_context,
                                                   pplcnet_params.resize_h,
                                                   pplcnet_params.resize_w,
                                                   pplcnet_params.pre_maxBatch,
                                                   pplcnet_params.pre_qsz,
                                                   pplcnet_params.infer_maxBatch,
                                                   pplcnet_params.custom_outputs,
                                                   pplcnet_params.onnx_file,
                                                   pplcnet_params.engine_file,
                                                   _ppyoloe_res_queue);
}

PipeLineProcessorTwoStage::~PipeLineProcessorTwoStage()
{
    _pplcnet_processor->StopModelThread();
    _ppyoloe_processor->StopModelThread();
    delete _pplcnet_processor;
    delete _ppyoloe_processor;

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

    delete _ppyoloe_res_queue;
    _ppyoloe_res_queue = nullptr;
}

int PipeLineProcessorTwoStage::InitVideoStreams(const int dev_id)
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

int PipeLineProcessorTwoStage::JpegEncode()
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

int PipeLineProcessorTwoStage::StartPipeline()
{
    // _video_queue->reset_count();
    this->_p_thread_jpeg = TaskThread(std::thread([this]() { this->JpegEncode(); }));
    _ppyoloe_processor->StartModelThread();
    _pplcnet_processor->StartModelThread();

    return 0;
}

std::vector<DecoderProcessorStats> PipeLineProcessorTwoStage::CollectVideoProcessorStats()
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

std::map<int, int> PipeLineProcessorTwoStage::ChcekDecoderStautsAndRestart()
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