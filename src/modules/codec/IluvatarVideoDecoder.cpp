//===----------------------------------------------------------------------===//
//
//   Copyright © 2022 Iluvatar CoreX. All rights reserved.
//   Copyright Declaration: This software, including all of its code and
//   documentation, except for the third-party software it contains, is a
//   copyrighted work of Shanghai Iluvatar CoreX Semiconductor Co., Ltd. and
//   its affiliates (“Iluvatar CoreX”) in accordance with the PRC Copyright Law
//   and relevant international treaties, and all rights contained therein are
//   enjoyed by Iluvatar CoreX. No user of this software shall have any right,
//   ownership or interest in this software and any use of this software shall
//   be in compliance with the terms and conditions of the End User License
//   Agreement.
//
//===----------------------------------------------------------------------===//
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include "nvviddec.h"
#include "IluvatarVideoDecoder.h"

using namespace std;

#define ALIGN_UP(val, alignment) ((((val) + (alignment)-1) / (alignment)) * (alignment))
#define ALIGN_DOWN(val, alignment) ((((val)) / (alignment)) * (alignment))
#define ALIGN_UP16(num) ALIGN_UP(num, 16)
#define ALIGN_UP32(num) ALIGN_UP(num, 32)

static auto ThrowOnCudaError = [](CUresult res, int lineNum = -1) {
    if (CUDA_SUCCESS != res)
    {
        stringstream ss;

        if (lineNum > 0)
        {
            ss << __FILE__ << ":";
            ss << lineNum << endl;
        }

        const char* errName = nullptr;
        if (CUDA_SUCCESS != cuGetErrorName(res, &errName))
        {
            ss << "CUDA error with code " << res << endl;
        }
        else
        {
            ss << "CUDA error: " << errName << endl;
        }

        const char* errDesc = nullptr;
        cuGetErrorString(res, &errDesc);

        if (!errDesc)
        {
            ss << "No error string available" << endl;
        }
        else
        {
            ss << errDesc << endl;
        }

        throw runtime_error(ss.str());
    }
};

struct Rect
{
    int l, t, r, b;
};

struct Dim
{
    int w, h;
};

namespace IX
{
struct IxDecoderImpl
{
    int id   = 0;
    int rate = -1;

    unsigned int frame_width = 0U, frame_height = 0U;

    CUstream   m_cuvidStream = nullptr;
    CUcontext  m_cuContext   = nullptr;
    CUvideodec m_hDecoder    = nullptr;

    int                   m_eCodec = 17;
    CUVIDDECODECREATEINFO m_cfg    = {0};
    std::string           fileName;

    atomic<int>    stop_error;
    atomic<int>    parser_error;
    atomic<int>    t_send_error;
    atomic<int>    t_get_error;
    atomic<size_t> m_nDecodedFrame;
    atomic<size_t> m_nFFSendFrame;

    TaskThread   thread_send;
    TaskThread   thread_get;
    atomic<bool> eos_set;
    atomic<bool> eos_send;
    atomic<bool> decoder_finish;
    atomic<bool> read_frame_fail;
};

int IluvatarVideoDecoder::GetCodec() const
{
    return p_impl->m_eCodec;
}

int IluvatarVideoDecoder::GetWidth()
{
    return p_impl->frame_width;
}

int IluvatarVideoDecoder::GetHeight()
{
    return p_impl->frame_height;
}

size_t IluvatarVideoDecoder::GetFrameSize()
{
    size_t const num_pixels = p_impl->frame_width * p_impl->frame_height * 3 / 2;

    return num_pixels;
}

size_t IluvatarVideoDecoder::GetDecoderTotalFrames()
{
    return p_impl->m_nDecodedFrame.load();
}

size_t IluvatarVideoDecoder::GetDecoderReceiveFrames()
{
    return p_impl->m_nFFSendFrame.load();
}

int IluvatarVideoDecoder::DecodeSatus()
{
    if (1 == p_impl->parser_error.load())
    {
        throw cuvid_parser_error("HandleVideoSource error!!!");
    }

    if (1 == p_impl->t_send_error.load())
    {
        throw decoder_error("SendBitStream thread error !!!");
    }

    if (1 == p_impl->t_get_error.load())
    {
        throw decoder_error("GetFrame2Queue thread error !!!");
    }

    if (p_impl->read_frame_fail.load())
    {
        logger->warn("[{} {}]: dec:{}, read frame fail \n", __FUNCTION__, __LINE__, p_impl->id);
        return -2;
    }

    if (p_impl->decoder_finish.load())
    {
        if (p_impl->eos_set.load())
            logger->warn("[{} {}]: dec:{}, frame all pop and eos get \n",
                        __FUNCTION__, __LINE__, p_impl->id);
        else
            logger->warn("[{} {}]: dec:{}, decoder finished (VPU error, no demuxer EOF) \n",
                        __FUNCTION__, __LINE__, p_impl->id);
        return 1;   // 无论 eos_set 是否为 true，都触发重启
    }

    return 0;
}

void IluvatarVideoDecoder::GetFrame2Queue(ProcessQueue<ViDecSurfaceCudaBuff>* m_DecFramesCtxQueue) noexcept
{
    p_impl->thread_get.threadId = std::this_thread::get_id();
    try
    {
        CUdeviceptr          cuDevPtr;
        CUVIDGETDECODESTATUS d_status;
        // uint64_t startTimestamp = GetCpuTimestamp();
        // uint64_t total_time = GetCpuTimestamp();
        // unsigned int index_10 = 0;
        CudaCtxPush ctxPush(p_impl->m_cuContext);
        while (!p_impl->thread_get.get_status())
        {
            uint64_t currentTimestamp = GetCpuTimestamp();

            CUVIDPROCPARAMS pVPP;
            unsigned int    nPitch = 0;
            auto            ret    = cuvidMapVideoFrame(p_impl->m_hDecoder, &cuDevPtr, &nPitch, &pVPP);

            if ((ret == CUDA_SUCCESS) && (nPitch > 0))
            {
                p_impl->m_nDecodedFrame++;
                // index_10++;
                // printf("%s %d, pVPP.h:%d\n",__FUNCTION__, __LINE__, pVPP.height);
                if (p_impl->m_nDecodedFrame.load() == 1)
                {
                    p_impl->frame_width  = nPitch;
                    p_impl->frame_height = pVPP.height;
                }
                if (p_impl->rate > 0)
                {
                    if ((p_impl->m_nDecodedFrame.load()) % p_impl->rate != 0)
                    {
                        ThrowOnCudaError(cuvidUnmapVideoFrame((p_impl->m_hDecoder), cuDevPtr, 0), __LINE__);
                        continue;
                    }
                }

                ViDecSurfaceCudaBuff frame(
                    nPitch, pVPP.height, YUV420, p_impl->m_cuContext, 1, p_impl->fileName, currentTimestamp);
                CUdeviceptr pFrame = frame.GetGpuMem();
                size_t      size   = frame.Total();

                if (p_impl->m_cuvidStream)
                {
                    ThrowOnCudaError(cuMemcpyDtoDAsync(pFrame, cuDevPtr, size, p_impl->m_cuvidStream), __LINE__);
                    CudaStrSync StrSync(p_impl->m_cuvidStream);
                }
                else
                {
                    ThrowOnCudaError(cuMemcpyDtoD(pFrame, cuDevPtr, size), __LINE__);
                    ThrowOnCudaError(cuCtxSynchronize(), __LINE__);
                }

                m_DecFramesCtxQueue->put(frame);
                ThrowOnCudaError(cuvidUnmapVideoFrame((p_impl->m_hDecoder), cuDevPtr, 0), __LINE__);
            }

            ThrowOnCudaError(cuvidGetDecodeStatus(p_impl->m_hDecoder, &d_status));
            if ((nPitch == 0) && (d_status.decodeStatus == cuvidDecodeStatus_Success))
            {
                p_impl->decoder_finish.store(true);
                logger->warn("[{} {}]: dec:{}, d_status.decodeStatus == cuvidDecodeStatus_Success \n",
                             __FUNCTION__,
                             __LINE__,
                             p_impl->id);
                break;
            }
        }
    }
    catch (exception& e)
    {
        std::cout << "rtsp" << p_impl->fileName << std::endl;
        cerr << e.what();
        p_impl->t_get_error.store(1);
    }
    logger->warn("[{} {}]: dec:{}, map frame thread exit \n", __FUNCTION__, __LINE__, p_impl->id);

    return;
}

void IluvatarVideoDecoder::sendEOF()
{
    if (p_impl->eos_send.load())
        return;

    size_t         send_times  = 0;
    CUVIDPICPARAMS picParam    = {0};
    picParam.pBitstreamData    = nullptr;
    picParam.nBitstreamDataLen = 0;
    picParam.timestamp         = 0;
    picParam.eos               = 1;

    while ((cuvidDecodePicture(p_impl->m_hDecoder, &picParam) == CUDA_ERROR_NOT_READY))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        send_times++;
        if (send_times >= 100)
        {
            ostringstream errorString;
            errorString << endl
                        << "Decoder id :" << p_impl->id << ", Send eof packet to decoder more than 100 !!!" << endl;
            throw runtime_error(errorString.str());
        }
    }
    logger->info("[{} {}]: dec:{}, send eof to decoder \n", __FUNCTION__, __LINE__, p_impl->id);
    p_impl->eos_send.store(true);
    return;
}

void IluvatarVideoDecoder::SendBitStream(FFmpegDemuxer* demuxer) noexcept
{
    p_impl->thread_send.threadId = std::this_thread::get_id();
    try
    {
        CUVIDGETDECODESTATUS d_status;
        CUresult             ret = CUDA_SUCCESS;

        uint8_t*       pVideo   = nullptr;
        PacketData     pkt_data = {0};
        CUVIDPICPARAMS picParam = {0};

        int         read_times = 0;
        CudaCtxPush ctxPush(p_impl->m_cuContext);

        while (!p_impl->thread_send.get_status())
        {
            size_t send_times  = 0;
            size_t pVideo_size = 0U;
            if (demuxer == nullptr)
            {
                throw runtime_error("FFmpeg Demux is Null !!!");
            }

            ThrowOnCudaError(cuvidGetDecodeStatus(p_impl->m_hDecoder, &d_status), __LINE__);
            if (d_status.decodeStatus == cuvidDecodeStatus_Success)
            {
                logger->warn("[{} {}]: dec:{}, d_status.decodeStatus == cuvidDecodeStatus_Success \n",
                             __FUNCTION__,
                             __LINE__,
                             p_impl->id);
                break;
            }

            if (!demuxer->Demux(pVideo, pVideo_size, pkt_data, nullptr, nullptr))
            {
                // std::this_thread::sleep_for(std::chrono::milliseconds(1));
                // continue;
                if (demuxer->IsEOF())
                    pVideo_size = 0U;
                else if (demuxer->IsConnectTimeout())
                {
                    p_impl->read_frame_fail.store(true);
                    logger->warn("[{} {}]: dec:{}, Connect Timeout \n", __FUNCTION__, __LINE__, p_impl->id);
                    break;
                }
                else
                {
                    ++read_times;
                    if ((read_times % 10) == 0)
                    {
                        logger->warn("[{} {}]: dec:{}, read frame fail and retry {} \n",
                                     __FUNCTION__,
                                     __LINE__,
                                     p_impl->id,
                                     read_times);
                        if (read_times > 1000)
                        {
                            p_impl->read_frame_fail.store(true);
                            break;
                        }
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }
            }
            read_times = 0;
            if ((pVideo_size > 0) && (pVideo != nullptr))
            {
                picParam.pBitstreamData    = pVideo;
                picParam.nBitstreamDataLen = pVideo_size;
                picParam.timestamp         = GetCpuTimestamp();
                picParam.eos               = 0;

                while ((cuvidDecodePicture(p_impl->m_hDecoder, &picParam) == CUDA_ERROR_NOT_READY) &
                       (!p_impl->thread_send.get_status()))
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    send_times++;
                    if (send_times >= 100)
                    {
                        ostringstream errorString;
                        errorString << endl
                                    << "Decoder id :" << p_impl->id << " , One Packet send times more than 100 !!!"
                                    << endl;
                        throw runtime_error(errorString.str());
                    }
                }
                p_impl->m_nFFSendFrame++;
            }
            else
            {
                p_impl->eos_set.store(true);
                logger->warn("[{} {}]: dec:{}, eof \n", __FUNCTION__, __LINE__, p_impl->id);
                break;
            }
        }
    }
    catch (exception& e)
    {
        cerr << e.what();
        p_impl->t_send_error.store(1);
    }
    sendEOF();
    logger->warn("[{} {}]: dec:{}, send stream thread exit \n", __FUNCTION__, __LINE__, p_impl->id);

    return;
}

int IluvatarVideoDecoder::HandleVideoSource(FFmpegDemuxer*                      demuxer,
                                            ProcessQueue<ViDecSurfaceCudaBuff>* m_DecFramesCtxQueue,
                                            int                                 rate) noexcept
{
    try
    {
        p_impl->stop_error.store(0);
        p_impl->parser_error.store(0);
        p_impl->t_send_error.store(0);
        p_impl->t_get_error.store(0);
        p_impl->m_nFFSendFrame.store(0);

        p_impl->eos_set.store(false);
        p_impl->eos_send.store(false);
        p_impl->decoder_finish.store(false);
        p_impl->read_frame_fail.store(false);

        p_impl->fileName = demuxer->GetSourceName();

        unsigned int demux_width  = demuxer->GetWidth();
        unsigned int demux_height = demuxer->GetHeight();

        double nb_frames = demuxer->GetAvgFramerate();
        if (rate <= 0)
            p_impl->rate = -1;
        else
        {
            p_impl->rate = int(nb_frames / rate);
            logger->info(
                "[{} {}]: dec:{}, nb_frames:{}, get {} frames", __FUNCTION__, __LINE__, p_impl->id, nb_frames, rate);
        }

        AVCodecID ff_Codec     = demuxer->GetVideoCodec();
        int       stream_codec = FFmpeg2IxCodecId(ff_Codec);

        if (p_impl->m_eCodec != stream_codec)
        {
            ostringstream errorString;
            errorString << endl
                        << "FFmpeg open source Codec:" << stream_codec << endl
                        << "Decoder Codec           :" << p_impl->m_eCodec << endl
                        << "FFmpeg open source Codec != Decoder Codec" << endl;
            throw runtime_error(errorString.str());
        }

        p_impl->thread_send = TaskThread(std::thread([this, demuxer]() { this->SendBitStream(demuxer); }));
        p_impl->thread_get =
            TaskThread(std::thread([this, m_DecFramesCtxQueue]() { this->GetFrame2Queue(m_DecFramesCtxQueue); }));
    }
    catch (exception& e)
    {
        cerr << e.what();
        p_impl->parser_error.store(1);
    }

    return 0;
}

int IluvatarVideoDecoder::StopVideoSource() noexcept
{
    try
    {
        p_impl->thread_send.set_status(true);
        p_impl->thread_send.join();

        sendEOF();

        p_impl->thread_get.set_status(true);
        p_impl->thread_get.join();
    }
    catch (exception& e)
    {
        cerr << e.what();
        p_impl->stop_error.store(1);
    }

    return 0;
}

void OnStreamChangedCallback(void* userData, CUVIDFormat* pFormat) {
    try
    {
        printf("Resolution changed to %dx%d\n", pFormat->picWidth, pFormat->picHeight);
    }
    catch (exception& e)
    {
        cerr << e.what();
    }
}

IluvatarVideoDecoder::IluvatarVideoDecoder(CUcontext cuContext, int eCodec, CUstream cuStream, int id)
{
    p_impl     = new IxDecoderImpl();
    p_impl->id = id;
    // if (cuStream == nullptr)
    // {
    //   CudaCtxPush ctxPush(cuContext);
    //   cudaStreamCreate(&(cuStream));
    // }
    p_impl->m_cuvidStream = cuStream;
    p_impl->m_cuContext   = cuContext;
    p_impl->m_eCodec      = eCodec;

    // p_impl->decode_error.store(0);
    p_impl->stop_error.store(0);
    p_impl->parser_error.store(0);
    p_impl->t_send_error.store(0);
    p_impl->t_get_error.store(0);

    CudaCtxPush ctxPush(p_impl->m_cuContext);

    p_impl->m_cfg.bitFormat       = eCodec;           // multiConfig->decConfig[i].stdMode;
    p_impl->m_cfg.bitstreamMode   = BS_MODE_PIC_END;  //(BitStreamMode)multiConfig->decConfig[i].bsmode;
    p_impl->m_cfg.wtlFormat       = 0;
    p_impl->m_cfg.scaleDownWidth  = 0;
    p_impl->m_cfg.scaleDownHeight = 0;

    checkCudaErrors(cuvidCreateDecoder(p_impl->m_cuContext, &(p_impl->m_hDecoder), &(p_impl->m_cfg)));

    CUVIDCALLBACK g_callback = { .pOnStreamChanged = OnStreamChangedCallback };

    checkCudaErrors(cuvidRegisterCallback(&(p_impl->m_hDecoder), nullptr, &g_callback));
}

IluvatarVideoDecoder::~IluvatarVideoDecoder()
{
    CudaCtxPush ctxPush(p_impl->m_cuContext);

    int ret = StopVideoSource();
    if (p_impl->m_hDecoder)
    {
        checkCudaErrors(cuvidDestroyDecoder(p_impl->m_hDecoder));
    }

    if (p_impl->m_cuvidStream != nullptr)
        checkCudaErrors(cudaStreamDestroy(p_impl->m_cuvidStream));

    delete p_impl;
}
}  // namespace IX
