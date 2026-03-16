#include "base_processor.h"
#include "json.h"

struct YoloV5PreSurfaces
{
    SurfaceCudaBuff                   preSurface;
    std::vector<ViDecSurfaceCudaBuff> oriSurfaces;
};

struct YoloV5Result
{
    ViDecSurfaceCudaBuff  oriSurfaces;
    std::vector<Rect>     rects;
    std::vector<RectInfo> rectsInfo;
};

class YoloV5ModelProcessor : public ModelProcessor
{
protected:
    ProcessQueue<ViDecSurfaceCudaBuff>* _video_queue;
    ProcessQueue<ViDecSurfaceCudaBuff>* _jpeg_queue;
    ProcessQueue<YoloV5Result>*         _res_queue;
    DetectPreprocessor*                 _m_preprocessor;
    ProcessQueue<YoloV5PreSurfaces>*    _m_preprocess_queue;

    virtual int InitPreProcess() override;
    virtual int InitIxrtIfner() override;

    virtual void IxrtInfer() override;
    virtual void CudaPreProcess() override;

public:
    YoloV5ModelProcessor(CUcontext                           cu_context,
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
                         ProcessQueue<YoloV5Result>*         res_queue);

    ~YoloV5ModelProcessor() override;

    virtual size_t GetPreQWaitNum() override { return _m_preprocess_queue->size(); }
    virtual size_t GetPreQDropNum() override { return _m_preprocess_queue->drops_count(); }
};

class PipeLineProcessorOneStage
{
private:
    int       _dev_id;
    CUcontext _cu_context;

    /* video */
    int                                  _dec_qsz = 0;
    std::vector<RtspUrlParams>           _rtsp_sources;
    std::map<int, VideoStreamProcessor*> _video_processores;
    std::map<int, int>                   _video_reset_count;
    ProcessQueue<ViDecSurfaceCudaBuff>*  _video_queue = nullptr;

    int InitVideoStreams(const int dev_id);

    /* jpeg */
    std::atomic<size_t>                 _p_jpegEncodeFrames = 0;
    int                                 _p_jpeg_maxBatch    = 0;
    int                                 _jpeg_qsz           = 0;
    IluvatarJpegCodec*                  _jpeg_encoder       = nullptr;
    ProcessQueue<ViDecSurfaceCudaBuff>* _jpeg_queue         = nullptr;
    TaskThread                          _p_thread_jpeg;

    int JpegEncode();

    /* YoloV5 */
    YoloV5ModelProcessor*       _yolov5_processor = nullptr;
    ProcessQueue<YoloV5Result>* _yolov5_res_queue = nullptr;

public:
    PipeLineProcessorOneStage(CUcontext                         cu_context,
                      const int                         dev_id,
                      const int                         jpeg_maxBatch,
                      const int                         jpeg_qsz,
                      const int                         dec_qsz,
                      const std::vector<RtspUrlParams>& rtsp_sources,
                      const ModelParams&                yolov5_params);

    size_t GetDecFrameNumber() { return _video_queue->count(); }
    size_t GetDecQWaitNum() { return _video_queue->size(); }
    size_t GetDecQDropNum() { return _video_queue->drops_count(); }

    size_t GetJpegFrameNumber() { return _p_jpegEncodeFrames.load(); }
    size_t GetJpegQWaitNum() { return _jpeg_queue->size(); }
    size_t GetJpegQDropNum() { return _jpeg_queue->drops_count(); }

    size_t GetYoloV5PreFrameNumber() { return _yolov5_processor->GetPreFrameNumber(); }
    size_t GetYoloV5InferFrameNumber() { return _yolov5_processor->GetInferFrameNumber(); }
    size_t GetYoloV5PreQWaitNum() { return _yolov5_processor->GetPreQWaitNum(); }
    size_t GetYoloV5PreQDropNum() { return _yolov5_processor->GetPreQDropNum(); }

    size_t GetYoloV5ResQWaitNum() { return _yolov5_res_queue->size(); }
    size_t GetYoloV5ResQDropNum() { return _yolov5_res_queue->drops_count(); }

    std::map<int, int>                 ChcekDecoderStautsAndRestart();
    std::vector<DecoderProcessorStats> CollectVideoProcessorStats();
    int                                CollectVideoProcessorNumbers() { return _video_processores.size(); }

    int StartPipeline();
    ~PipeLineProcessorOneStage();
};