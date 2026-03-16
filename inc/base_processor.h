#include "FFmpegDemuxer.h"
#include "IluvatarJpegCodec.h"
#include "IluvatarPreprocess.h"
#include "memoryInterface.h"
#include "IluvatarIxRT.h"
#include "IluvatarVideoDecoder.h"
using VideoDecoder = IX::IluvatarVideoDecoder;

#include "util.h"

enum ProcessorStatus
{
    DEC_WORK               = 0,
    DEC_FINISH             = 1,
    FFMPEG_READ_FRAME_FAIL = -2,
};

struct DecoderProcessorStats
{
    int    id;
    size_t mapFramesOK     = 0;
    size_t receiveFramesOK = 0;
};

struct RectInfo
{
    int   label;
    float score;
};

class VideoStreamProcessor
{
private:
    CUcontext   _cuContext;
    std::string _input_file;
    int         _id;

    FFmpegDemuxer*                      demuxer;
    VideoDecoder*                       decoder;
    ProcessQueue<ViDecSurfaceCudaBuff>* _frame_queue;

    size_t          count   = 0;
    int             width   = 0;
    int             height  = 0;
    std::string     codec   = "None";
    ProcessorStatus _status = DEC_WORK;

public:
    VideoStreamProcessor(CUcontext                           cuContext,
                         const int                           dev_id,
                         const std::string&                  input_file,
                         ProcessQueue<ViDecSurfaceCudaBuff>* frame_queue,
                         int                                 id,
                         int                                 rate        = -1,
                         bool                                enable_jenc = true,
                         bool                                enable_jdec = false);
    ~VideoStreamProcessor();

    ProcessorStatus GetStatus();
    int             GetDecStatus();
    std::string     GetInputFileName();

    size_t GetDecoderMapFramesOK();
    size_t GetDecoderReceiveFramesOK();
};

template <typename T>
static int waitEnoughBatch(ProcessQueue<T>* queue, int batch, int delay_time, bool exit_signal)
{
    int timeout = 0;
    int t_batch = 0;
    while (timeout < 100)
    {
        if (exit_signal)
        {
            return 0;
        }
        t_batch = queue->size();
        if (t_batch >= batch)
            return batch;
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_time));

        ++timeout;
    }
    return t_batch;
}

class ModelProcessor
{
protected:
    CUcontext _m_cu_context;
    int       _m_pre_qsz        = 0;
    int       _m_resize_h       = 0;
    int       _m_resize_w       = 0;
    int       _m_pre_maxBatch   = 1;
    int       _m_infer_maxBatch = 1;
    Trt*      _m_infer_model;

    cudaStream_t _m_infer_cudastream;

    size_t _m_model_width  = 0;
    size_t _m_model_height = 0;

    std::string              _m_onnx_file;
    std::string              _m_engine_file;
    std::vector<std::string> _m_input_files;
    std::vector<std::string> _m_model_outputs;
    std::atomic<size_t>      _m_PreproFrames;
    std::atomic<size_t>      _m_InferFrames;
    TaskThread               _m_thread_preprocess;
    TaskThread               _m_thread_infer;

    virtual int InitPreProcess() { return 0; }
    virtual int InitIxrtIfner() { return 0; }

    virtual void IxrtInfer() { return; }
    virtual void CudaPreProcess() { return; }

public:
    ModelProcessor(CUcontext                       cu_context,
                   int                             resize_h,
                   int                             resize_w,
                   int                             pre_maxBatch,
                   int                             pre_qsz,
                   int                             infer_maxBatch,
                   const std::vector<std::string>& custom_outputs,
                   const std::string&              onnx_file,
                   const std::string&              engine_file);

    virtual size_t GetPreQWaitNum() = 0;
    virtual size_t GetPreQDropNum() = 0;
    size_t         GetPreFrameNumber() { return _m_PreproFrames.load(); }
    size_t         GetInferFrameNumber() { return _m_InferFrames.load(); }

    int InitModelHandle();
    int StartModelThread();
    int StopModelThread();
    virtual ~ModelProcessor();
};