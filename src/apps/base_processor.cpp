#include "base_processor.h"
using namespace std;
using VideoDecoder = IX::IluvatarVideoDecoder;

static std::string CodecId2String(int codec)
{
    switch (codec)
    {
        case 0: return "H264";   // 0 STD_AVC
        case 12: return "H265";  // 12 STD_HEVC
        case 13: return "VP9";   // 13 STD_VP9
        default: return "None";  // 17
    }
}

VideoStreamProcessor::VideoStreamProcessor(CUcontext                           cuContext,
                                           const int                           dev_id,
                                           const std::string&                  input_file,
                                           ProcessQueue<ViDecSurfaceCudaBuff>* frame_queue,
                                           int                                 id,
                                           int                                 rate,
                                           bool                                enable_jenc,
                                           bool                                enable_jdec)
    : _cuContext(cuContext)
    , _input_file(input_file)
    , _id(id)
    , _frame_queue(frame_queue)
{
    std::map<std::string, std::string> ffmpeg_options = {};
    if (strstr(_input_file.c_str(), "rtsp://") != NULL)  // option for rtsp stream
    {
        ffmpeg_options = {{"rtsp_transport", "tcp"},
                          {"probesize", "500000000"},
                          {"analyzeduration", "500000000"},
                          {"buffer_size", "1024000"},
                          {"timeout", "2000000"},
                          {"stimeout", "2000000"},
                          {"max_delay", "1000000"},
                          {"rw_timeout", "1000000"}};
    }
    else if (strstr(_input_file.c_str(), "udp://") != NULL)  // option for udp stream
    {
        ffmpeg_options = {{"buffer_size",      "10485760"},  // 10MB UDP socket recv buffer for 4K bitrate
                          {"fifo_size",         "10000000"},  // FFmpeg internal FIFO to absorb bursts
                          {"overrun_nonfatal",  "1"},         // drop frames on FIFO overflow instead of exiting
                          {"reuse",             "1"}};
    }

    demuxer             = new FFmpegDemuxer(_input_file.c_str(), ffmpeg_options);
    AVCodecID ff_Codec  = demuxer->GetVideoCodec();
    int       dec_codec = FFmpeg2IxCodecId(ff_Codec);

    decoder = new VideoDecoder(_cuContext, dec_codec, nullptr, _id);
    decoder->Init(demuxer, _frame_queue, rate);

    codec       = CodecId2String(dec_codec);
    int width_  = demuxer->GetWidth();
    int height_ = demuxer->GetHeight();
    logger->info(
        "[{} {}]: id {}:{} dmux width:{}, dmux height:{} \n", __FUNCTION__, __LINE__, id, input_file, width_, height_);
}

VideoStreamProcessor::~VideoStreamProcessor()
{
    if (decoder != nullptr)
    {
        delete decoder;
        decoder = nullptr;
    }

    if (demuxer != nullptr)
    {
        delete demuxer;
        demuxer = nullptr;
    }
}

ProcessorStatus VideoStreamProcessor::GetStatus()
{
    return _status;
}

int VideoStreamProcessor::GetDecStatus()
{
    return decoder->DecodeSatus();
}

size_t VideoStreamProcessor::GetDecoderMapFramesOK()
{
    return decoder->GetDecoderTotalFrames();
}

size_t VideoStreamProcessor::GetDecoderReceiveFramesOK()
{
    return decoder->GetDecoderReceiveFrames();
}

std::string VideoStreamProcessor::GetInputFileName()
{
    return _input_file;
}

ModelProcessor::ModelProcessor(CUcontext                       cu_context,
                               int                             resize_h,
                               int                             resize_w,
                               int                             pre_maxBatch,
                               int                             pre_qsz,
                               int                             infer_maxBatch,
                               const std::vector<std::string>& custom_outputs,
                               const std::string&              onnx_file,
                               const std::string&              engine_file)
    : _m_cu_context(cu_context)
    , _m_onnx_file(onnx_file)
    , _m_engine_file(engine_file)
    , _m_resize_h(resize_h)
    , _m_resize_w(resize_w)
    , _m_pre_qsz(pre_qsz)
    , _m_infer_maxBatch(infer_maxBatch)
    , _m_pre_maxBatch(pre_maxBatch)
    , _m_model_outputs(custom_outputs)
{
    CudaCtxPush ctxPush(_m_cu_context);
    checkCudaErrors(cudaStreamCreate(&_m_infer_cudastream));
}

ModelProcessor::~ModelProcessor()
{
    CudaCtxPush ctxPush(_m_cu_context);
    if (_m_infer_cudastream)
    {
        checkCudaErrors(cudaStreamDestroy(_m_infer_cudastream));
    }
}

int ModelProcessor::InitModelHandle()
{
    CudaCtxPush ctxPush(_m_cu_context);
    InitPreProcess();
    InitIxrtIfner();

    return 0;
}

int ModelProcessor::StartModelThread()
{
    this->_m_thread_preprocess = TaskThread(std::thread([this]() { this->CudaPreProcess(); }));
    this->_m_thread_infer      = TaskThread(std::thread([this]() { this->IxrtInfer(); }));

    return 0;
}

int ModelProcessor::StopModelThread()
{
    this->_m_thread_infer.set_status(true);
    this->_m_thread_infer.join();

    this->_m_thread_preprocess.set_status(true);
    this->_m_thread_preprocess.join();

    return 0;
}