#include "IluvatarPreprocess.h"
#include "preprocess.h"
using namespace std;

struct PreProcessImpl
{
    cudaStream_t stream;
    CUcontext    ctx_      = nullptr;
    int          resize_h_ = 0;
    int          resize_w_ = 0;
    int          maxinput_ = 0;
    float*       d_scale   = nullptr;
    float*       d_base    = nullptr;

    uint8_t**                    crprz_d_src      = nullptr;
    int2*                        crprz_d_srcShape = nullptr;
    iluvatar::cropResize::RectA* d_rect           = nullptr;
    int                          rects_size       = 0;

    // Pre-allocated intermediate buffer for CropResize output.
    // Safe to reuse every frame: both CropResize and CvtNorm run on the same
    // stream, so the GPU serializes them without a host-side sync in between.
    SurfaceCudaBuff cached_resize;
};

template <typename T>
static bool CropResizeYuv420P(std::vector<T>&                 srcImages,
                              std::vector<std::vector<Rect>>& rects,
                              SurfaceCudaBuff&                dstImage,
                              cudaStream_t                    stream,
                              uint8_t**                       d_src,
                              int2*                           d_srcShape,
                              iluvatar::cropResize::RectA*    d_rect,
                              const int                       maxRectSize)
{
    if (srcImages.size() != rects.size())
    {
        logger->error("[{} {}]: input images number {} != rects size {} \n",
                      __FUNCTION__,
                      __LINE__,
                      srcImages.size(),
                      rects.size());
        return false;
    }

    for (size_t i = 0; i < srcImages.size(); i++)
    {
        if (srcImages[i].GetPixelFormat() != YUV420)
        {
            logger->error("[{} {}]: only support yuv420 input in faster kernel !!!\n", __FUNCTION__, __LINE__);
            return false;
        }
    }

    int numberOfImages = srcImages.size();

    /* total rect numbers */
    int totalRectNum = 0;
    for (size_t i = 0; i < rects.size(); i++)
    {
        totalRectNum += rects[i].size();
    }

    /* rect to RectA */
    std::vector<iluvatar::cropResize::RectA> vRect(totalRectNum);
    int                                      idx = 0;
    for (unsigned int i = 0; i < rects.size(); i++)
    {
        for (size_t j = 0; j < rects[i].size(); j++)
        {
            vRect[idx] = {i, {rects[i][j].x, rects[i][j].y, rects[i][j].width, rects[i][j].height}};
            idx++;
        }
    }

    if (vRect.size() > maxRectSize)
    {
        logger->error(
            "[{} {}]: real rects size {} > kernel limited {}!!!\n", __FUNCTION__, __LINE__, vRect.size(), maxRectSize);
        return false;
    }
    /* rect data to device */
    size_t rectBufSize = vRect.size() * sizeof(iluvatar::cropResize::RectA);
    checkCudaErrors(cudaMemcpyAsync(d_rect, vRect.data(), rectBufSize, cudaMemcpyHostToDevice, stream));
    uint8_t* d_dst = reinterpret_cast<uint8_t*>(dstImage.GetGpuMem());

    std::ostringstream oss;
    oss << "d_src:" << d_src <<", d_dst:" << static_cast<void*>(d_dst) << ", d_srcShape:" << static_cast<void*>(d_srcShape) << ", d_rect:" << static_cast<void*>(d_rect);
    std::vector<uint8_t*> d_src_vec(numberOfImages, nullptr);
    std::vector<int2>     h_srcShape(numberOfImages);
    for (size_t i = 0; i < numberOfImages; i++)
    {
        /* image data */
        d_src_vec[i] = reinterpret_cast<uint8_t*>(srcImages[i].GetGpuMem());
        oss << ", {d_src[" << i << "]:" << static_cast<void*>(d_src_vec[i]);
        /* image shape */
        h_srcShape[i] = {srcImages[i].GetWidth(), srcImages[i].GetHeight()};
        oss << ", d_shape[" << i << "]:(" << srcImages[i].GetWidth() << ";" << srcImages[i].GetHeight() << ")";
        for (size_t j = 0; j < rects[i].size(); j++)
        {
            oss << ", rect[" << j << "]:(" << rects[i][j].x << ";" << rects[i][j].y << ";" << rects[i][j].width << ";"
                << rects[i][j].height << ")";
        }
        oss << "}";
    }
    logger->debug("[{} {}]:{}", __FUNCTION__, __LINE__, oss.str());

    size_t srcBatchBuffer = numberOfImages * sizeof(uint8_t*);
    size_t srcShapeBuffer = numberOfImages * sizeof(int2);

    checkCudaErrors(cudaMemcpyAsync(d_src, d_src_vec.data(), srcBatchBuffer, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_srcShape, h_srcShape.data(), srcShapeBuffer, cudaMemcpyHostToDevice, stream));

    size_t resize_w = dstImage.GetWidth();
    size_t resize_h = dstImage.GetHeight();

    iluvatar::cropResize::CropResize(d_src,
                                     d_dst,
                                     d_srcShape,
                                     resize_w,
                                     resize_h,
                                     d_rect,
                                     totalRectNum,
                                     iluvatar::cropResize::CropResizeColorType::CROP_RESIZE_IYUV,
                                     iluvatar::cropResize::CropResizeInterType::INTERP_LINEAR,
                                     stream);

    checkCudaErrors(cudaStreamSynchronize(stream));
    return true;
}

static bool CvtcolorConvertoNormalizeReformat(SurfaceCudaBuff& srcImage,
                                              SurfaceCudaBuff& dstImage,
                                              cudaStream_t     stream,
                                              float*           d_scale = nullptr,
                                              float*           d_base  = nullptr)
{
    if (srcImage.GetPixelFormat() != YUV420)
    {
        logger->error("[{} {}]: only support yuv420 input in faster kernel !!!\n", __FUNCTION__, __LINE__);
        return false;
    }

    int      batches   = srcImage.GetBatch();
    size_t   srcWidth  = srcImage.GetWidth();
    size_t   srcHeight = srcImage.GetHeight();
    uint8_t* srcData   = reinterpret_cast<uint8_t*>(srcImage.GetGpuMem());
    float*   dstData   = reinterpret_cast<float*>(dstImage.GetGpuMem());

    float alpha = 1.0f / 255.f;
    float beta  = 0.0f;

    int3 base_size  = {1, 1, 1};
    int3 scale_size = {1, 1, 1};

    float    globalScale = 1.f;
    float    globalShift = 0.f;
    float    epsilon     = 0.f;
    uint32_t flags       = NORMALIZE_SCALE_IS_STDDEV;

    bool norm_flag = d_scale == nullptr ? false : true;

    iluvatar::CvtNormReformat::CvtNormReformat(
        srcData,
        dstData,
        iluvatar::CvtNormReformat::CvtColorReformatType::CVTCOLOR_REFORMAT_IYUV2RGBf32p,
        batches,
        srcWidth,
        srcHeight,
        alpha,
        beta,
        norm_flag,
        d_base,
        base_size,
        d_scale,
        scale_size,
        globalScale,
        globalShift,
        epsilon,
        flags,
        stream);

    checkCudaErrors(cudaStreamSynchronize(stream));
    return true;
}

DetectPreprocessor::DetectPreprocessor(int resize_h, int resize_w, CUcontext context, int maxInputs)
{
    p_impl       = new PreProcessImpl();
    p_impl->ctx_ = context;

    p_impl->resize_h_ = resize_h;
    p_impl->resize_w_ = resize_w;

    p_impl->maxinput_  = maxInputs;
    p_impl->rects_size = maxInputs;

    checkCudaErrors(cudaStreamCreate(&(p_impl->stream)));

    size_t srcBatchBuffer = maxInputs * sizeof(uint8_t*);
    size_t srcShapeBuffer = maxInputs * sizeof(int2);
    checkCudaErrors(cudaMalloc((void**)&(p_impl->crprz_d_src), srcBatchBuffer));
    checkCudaErrors(cudaMalloc((void**)&(p_impl->crprz_d_srcShape), srcShapeBuffer));

    size_t rectBufSize = maxInputs * sizeof(iluvatar::cropResize::RectA);
    checkCudaErrors(cudaMalloc((void**)&(p_impl->d_rect), rectBufSize));

    // Pre-allocate the intermediate YUV buffer so Process() is allocation-free.
    p_impl->cached_resize = SurfaceCudaBuff(
        resize_w, resize_h, YUV420, context, maxInputs, Buffer::Flag::PPYOLOE_CROP_RESIZE_FLAG);
}

DetectPreprocessor::~DetectPreprocessor()
{
    if (p_impl->stream)
    {
        checkCudaErrors(cudaStreamSynchronize(p_impl->stream));
    }

    checkCudaErrors(cudaFree(p_impl->crprz_d_src));
    checkCudaErrors(cudaFree(p_impl->d_rect));
    checkCudaErrors(cudaFree(p_impl->crprz_d_srcShape));
    p_impl->crprz_d_src      = nullptr;
    p_impl->crprz_d_srcShape = nullptr;
    p_impl->d_rect           = nullptr;

    if (p_impl->stream)
    {
        checkCudaErrors(cudaStreamDestroy(p_impl->stream));
    }

    delete p_impl;
    p_impl = nullptr;
}

SurfaceCudaBuff DetectPreprocessor::Process(std::vector<ViDecSurfaceCudaBuff>& inputImage)
{
    // auto start_time1 = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < inputImage.size(); i++)
    {
        if (inputImage[i].Empty())
        {
            ostringstream errorString;
            errorString << endl << __FUNCTION__ << " " << __LINE__ << ": input [" << i << "]: is Empty !!!" << endl;
            throw runtime_error(errorString.str());
        }

        if (inputImage[i].GetPixelFormat() != YUV420)
        {
            logger->error("[{} {}]: only support yuv420 input in faster kernel !!!\n", __FUNCTION__, __LINE__);
            return SurfaceCudaBuff();
        }

        if (inputImage[i].GetBatch() != 1)
        {
            logger->error("[{} {}]: vector {}:this surface batch != 1 !!!\n", __FUNCTION__, __LINE__, i);
            return SurfaceCudaBuff();
        }
    }

    int batch = inputImage.size();
    if (batch > p_impl->maxinput_)
    {
        logger->error("[{} {}]: batch {} > limit batch {} !!!\n", __FUNCTION__, __LINE__, batch, p_impl->maxinput_);
        return SurfaceCudaBuff();
    }
    // yuv resize — use pre-allocated intermediate buffer (no cudaMalloc in hot path)
    std::vector<std::vector<Rect>> rects;
    rects.reserve(inputImage.size());
    for (size_t i = 0; i < inputImage.size(); i++)
    {
        rects.push_back({{0, 0, int(inputImage[i].GetWidth()), int(inputImage[i].GetHeight())}});
    }

    if (!CropResizeYuv420P(inputImage,
                           rects,
                           p_impl->cached_resize,
                           p_impl->stream,
                           p_impl->crprz_d_src,
                           p_impl->crprz_d_srcShape,
                           p_impl->d_rect,
                           p_impl->rects_size))
    {
        logger->error("[{} {}]: yuv resize fail !!!\n", __FUNCTION__, __LINE__);
        return SurfaceCudaBuff();
    }

    // yuv2rgb + 1/255 normalize + reformat
    // NOTE: CropResize and CvtNorm are on the same stream, so the GPU guarantees
    // sequential execution without a host-side sync in between.
    SurfaceCudaBuff outputImage(p_impl->resize_w_, p_impl->resize_h_, RGB_32F_PLANAR, p_impl->ctx_, batch);
    if (!CvtcolorConvertoNormalizeReformat(p_impl->cached_resize, outputImage, p_impl->stream))
    {
        logger->error("[{} {}]: yuv faster kernel fail !!!\n", __FUNCTION__, __LINE__);
        return SurfaceCudaBuff();
    }

    return outputImage;
}

ClassifyPreprocessor::ClassifyPreprocessor(int resize_h, int resize_w, CUcontext context, int maxInputs)
{
    p_impl       = new PreProcessImpl();
    p_impl->ctx_ = context;

    p_impl->resize_h_ = resize_h;
    p_impl->resize_w_ = resize_w;

    p_impl->maxinput_ = maxInputs;

    checkCudaErrors(cudaStreamCreate(&(p_impl->stream)));

    std::vector<float> baseVec  = {0.485f, 0.456f, 0.406f};
    std::vector<float> scaleVec = {0.229, 0.224, 0.225};

    size_t scaleBufSize = scaleVec.size() * sizeof(float);
    size_t baseBufSize  = baseVec.size() * sizeof(float);
    checkCudaErrors(cudaMalloc((void**)&(p_impl->d_scale), scaleBufSize));
    checkCudaErrors(cudaMalloc((void**)&(p_impl->d_base), baseBufSize));
    checkCudaErrors(cudaMemcpy((p_impl->d_scale), scaleVec.data(), scaleBufSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((p_impl->d_base), baseVec.data(), baseBufSize, cudaMemcpyHostToDevice));

    size_t srcBatchBuffer = maxInputs * sizeof(uint8_t*);
    size_t srcShapeBuffer = maxInputs * sizeof(int2);
    checkCudaErrors(cudaMalloc((void**)&(p_impl->crprz_d_src), srcBatchBuffer));
    checkCudaErrors(cudaMalloc((void**)&(p_impl->crprz_d_srcShape), srcShapeBuffer));

    size_t rectBufSize = maxInputs * 100 * sizeof(iluvatar::cropResize::RectA);
    checkCudaErrors(cudaMalloc((void**)&(p_impl->d_rect), rectBufSize));
    p_impl->rects_size = maxInputs * 100;
}

ClassifyPreprocessor::~ClassifyPreprocessor()
{
    if (p_impl->stream)
    {
        checkCudaErrors(cudaStreamSynchronize(p_impl->stream));
    }

    checkCudaErrors(cudaFree(p_impl->crprz_d_src));
    checkCudaErrors(cudaFree(p_impl->d_rect));
    checkCudaErrors(cudaFree(p_impl->crprz_d_srcShape));
    p_impl->crprz_d_src      = nullptr;
    p_impl->crprz_d_srcShape = nullptr;
    p_impl->d_rect           = nullptr;

    checkCudaErrors(cudaFree(p_impl->d_base));
    checkCudaErrors(cudaFree(p_impl->d_scale));
    p_impl->d_scale = nullptr;
    p_impl->d_base  = nullptr;

    if (p_impl->stream)
    {
        checkCudaErrors(cudaStreamDestroy(p_impl->stream));
    }

    delete p_impl;
    p_impl = nullptr;
}

SurfaceCudaBuff ClassifyPreprocessor::Process(std::vector<ViDecSurfaceCudaBuff>& inputImage,
                                             std::vector<std::vector<Rect>>&    rects)
{
    for (size_t i = 0; i < inputImage.size(); i++)
    {
        if (inputImage[i].Empty())
        {
            ostringstream errorString;
            errorString << endl << __FUNCTION__ << " " << __LINE__ << ": input [" << i << "]: is Empty !!!" << endl;
            throw runtime_error(errorString.str());
        }

        if (inputImage[i].GetBatch() != 1)
        {
            logger->error("[{} {}]: vector {}:this surface batch != 1 !!!\n", __FUNCTION__, __LINE__, i);
            return SurfaceCudaBuff();
        }
    }

    if (inputImage.size() > p_impl->maxinput_)
    {
        logger->error(
            "[{} {}]: batch {} > limit batch {} !!!\n", __FUNCTION__, __LINE__, inputImage.size(), p_impl->maxinput_);
        return SurfaceCudaBuff();
    }

    int batch = 0;
    for (size_t i = 0; i < rects.size(); i++)
    {
        batch += rects[i].size();
    }

    // // yuv resize
    SurfaceCudaBuff cprzImage(
        p_impl->resize_w_, p_impl->resize_h_, YUV420, p_impl->ctx_, batch, Buffer::Flag::PPLCNET_CROP_RESIZE_FLAG);
    if (!CropResizeYuv420P(inputImage,
                           rects,
                           cprzImage,
                           p_impl->stream,
                           p_impl->crprz_d_src,
                           p_impl->crprz_d_srcShape,
                           p_impl->d_rect,
                           p_impl->rects_size))
    {
        logger->error("[{} {}]: yuv resize fail !!!\n", __FUNCTION__, __LINE__);
        return SurfaceCudaBuff();
    }

    // // yuv2rgb 1/255 normalize reformat
    SurfaceCudaBuff outputImage(p_impl->resize_w_, p_impl->resize_h_, RGB_32F_PLANAR, p_impl->ctx_, batch);
    if (!CvtcolorConvertoNormalizeReformat(cprzImage, outputImage, p_impl->stream, p_impl->d_scale, p_impl->d_base))
    {
        logger->error("[{} {}]: yuv faster kernel fail !!!\n", __FUNCTION__, __LINE__);
        return SurfaceCudaBuff();
    }

    return outputImage;
}