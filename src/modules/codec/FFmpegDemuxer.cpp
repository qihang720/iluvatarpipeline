/*
 * Copyright 2019 NVIDIA Corporation
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *    http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "FFmpegDemuxer.h"
#include <cassert>
#include <cerrno>
#include <iostream>
#include <limits>
#include <sstream>
#include "libavutil/avstring.h"
#include "libavutil/avutil.h"
#include <unistd.h>

using namespace std;

#define LIBAVFORMAT_INTERRUPT_OPEN_TIMEOUT_MS 30000
#define LIBAVFORMAT_INTERRUPT_READ_TIMEOUT_MS 30000

static string AvErrorToString(int av_error_code)
{
    const auto buf_size   = 1024U;
    char*      err_string = (char*)calloc(buf_size, sizeof(*err_string));
    if (!err_string)
    {
        return string();
    }

    if (0 != av_strerror(av_error_code, err_string, buf_size - 1))
    {
        free(err_string);
        stringstream ss;
        ss << "Unknown error with code " << av_error_code;
        return ss.str();
    }

    string str(err_string);
    free(err_string);
    return str;
}

int DataProvider::GetData(uint8_t* pBuf, int nBuf)
{
    if (i_str.eof())
    {
        return AVERROR_EOF;
    }

    if (!i_str.good())
    {
        return AVERROR_UNKNOWN;
    }

    try
    {
        i_str.read((char*)pBuf, nBuf);
        return i_str.gcount();
    }
    catch (exception& e)
    {
        cerr << e.what() << endl;
        return AVERROR_UNKNOWN;
    }
}

static inline void get_monotonic_time(timespec* time)
{
    clock_gettime(CLOCK_MONOTONIC, time);
}

static inline timespec get_monotonic_time_diff(timespec start, timespec end)
{
    timespec temp;
    if (end.tv_nsec - start.tv_nsec < 0)
    {
        temp.tv_sec  = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }
    else
    {
        temp.tv_sec  = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp;
}

static inline double get_monotonic_time_diff_ms(timespec time1, timespec time2)
{
    timespec delta        = get_monotonic_time_diff(time1, time2);
    double   milliseconds = delta.tv_sec * 1000 + (double)delta.tv_nsec / 1000000.0;

    return milliseconds;
}

static inline int _opencv_ffmpeg_interrupt_callback(void* ptr)
{
    AVInterruptCallbackMetadata* metadata = (AVInterruptCallbackMetadata*)ptr;
    assert(metadata);

    if (metadata->timeout_after_ms == 0)
    {
        return 0;  // timeout is disabled
    }

    timespec now;
    get_monotonic_time(&now);

    metadata->timeout = get_monotonic_time_diff_ms(metadata->value, now) > metadata->timeout_after_ms;

    return metadata->timeout ? -1 : 0;
}

static inline int check_i_frame(char *data, int size, AVCodecID bitFormat)
{
    if ((size <= 4) || (data == NULL))
        return -1;

    if (bitFormat == AV_CODEC_ID_AVS2)
    {
        size_t pos = 0;
        const uint8_t *u8data = (const uint8_t *)data;
        while (pos + 3 < (size_t)size) {
            if ((u8data[pos] == 0x00 && u8data[pos+1] == 0x00 &&
                (u8data[pos+2] == 0x01 || (u8data[pos+2] >= 0xB0 && u8data[pos+2] <= 0xB6)))) {

                size_t nal_start;
                if (u8data[pos+2] == 0x01)
                    nal_start = pos + 3;
                else
                    nal_start = pos + 2;

                uint8_t nal_type = u8data[nal_start];

                if (nal_type == 0xB0)
                    return 1;  
                if (nal_type == 0xB3)
                    return 2;   
                if (nal_type == 0xB6)
                    return 0;   
            }
            pos++;
        }
        return -1; 
    }

    unsigned int pos = 0;
    unsigned int k = 0;
    unsigned int nalUnitType = 0;
    bool bStartCode = false;
    char *scData = data;

    while (k < (size - 4))
    {
        if ((0 == scData[k]) && (0 == scData[k+1]) && (1 == scData[k+2]))
        {
            pos = k + 3;
            bStartCode = true;
        }
        else if ((0 == scData[k]) && (0 == scData[k+1]) && (0 == scData[k+2]) && (1 == scData[k+3]))
        {
            pos = k + 4;
            bStartCode = true;
        }

        if (bStartCode)
        {
            bStartCode = false;
            if (bitFormat == AV_CODEC_ID_H264)
            {
                nalUnitType = scData[pos] & 0x1f;
                if ((nalUnitType == 0x5) || (nalUnitType == 0x7))
                    return nalUnitType;
            }
            else if (bitFormat == AV_CODEC_ID_HEVC)
            {
                nalUnitType = (scData[pos] & 0x7E) >> 1;
                if (((nalUnitType >= 0x10) && (nalUnitType <= 0x15)) || (nalUnitType == 0x20))
                    return nalUnitType;
            }
        }
        k++;
        usleep(1);
    }

    return -1;
}



DataProvider::DataProvider(std::istream& istr)
    : i_str(istr)
{
}

FFmpegDemuxer::FFmpegDemuxer(const char* szFilePath, const map<string, string>& ffmpeg_options)
    : FFmpegDemuxer(CreateFormatContext(szFilePath, ffmpeg_options))
{
    szFile = szFilePath;
    if (strcmp(szFilePath, "rtsp") && strcmp(szFilePath, "rtmp"))
        is_rtsp = true;
}

FFmpegDemuxer::FFmpegDemuxer(DataProvider& pDataProvider, const map<string, string>& ffmpeg_options)
    : FFmpegDemuxer(CreateFormatContext(pDataProvider, ffmpeg_options))
{
    avioc = fmtc->pb;
}

uint32_t FFmpegDemuxer::GetWidth() const
{
    return width;
}

uint32_t FFmpegDemuxer::GetHeight() const
{
    return height;
}

uint32_t FFmpegDemuxer::GetGopSize() const
{
    return gop_size;
}

uint32_t FFmpegDemuxer::GetNumFrames() const
{
    return nb_frames;
}

double FFmpegDemuxer::GetFramerate() const
{
    return framerate;
}

double FFmpegDemuxer::GetAvgFramerate() const
{
    return avg_framerate;
}

double FFmpegDemuxer::GetTimebase() const
{
    return timebase;
}

std::string FFmpegDemuxer::GetSourceName() const
{
    return szFile;
}

bool FFmpegDemuxer::IsVFR() const
{
    return framerate != avg_framerate;
}

bool FFmpegDemuxer::IsEOF() const
{
    return is_EOF;
}

bool FFmpegDemuxer::IsConnectTimeout() const
{
    return is_ConnectTimeout;
}

uint32_t FFmpegDemuxer::GetVideoStreamIndex() const
{
    return videoStream;
}

AVPixelFormat FFmpegDemuxer::GetPixelFormat() const
{
    return eChromaFormat;
}

AVColorSpace FFmpegDemuxer::GetColorSpace() const
{
    return color_space;
}

AVColorRange FFmpegDemuxer::GetColorRange() const
{
    return color_range;
}

extern unsigned long GetNumDecodeSurfaces(cudaVideoCodec eCodec, unsigned int nWidth, unsigned int nHeight);


bool FFmpegDemuxer::Demux(uint8_t*&   pVideo,
                          size_t&     rVideoBytes,
                          PacketData& pktData,
                          uint8_t**   ppSEI,
                          size_t*     pSEIBytes)
{
    if (!fmtc)
    {
        return false;
    }

    if (pktSrc.data)
    {
        av_packet_unref(&pktSrc);
    }

    if (!annexbBytes.empty())
    {
        annexbBytes.clear();
    }

    if (!seiBytes.empty())
    {
        seiBytes.clear();
    }

    auto appendBytes = [](vector<uint8_t>& elementaryBytes,
                          AVPacket&        avPacket,
                          AVPacket&        avPacketOut,
                          AVBSFContext*    pAvbsfContext,
                          int              streamId,
                          bool             isFilteringNeeded)
    {
        if (avPacket.stream_index != streamId)
        {
            return;
        }

        if (isFilteringNeeded)
        {
            if (avPacketOut.data)
            {
                av_packet_unref(&avPacketOut);
            }

            av_bsf_send_packet(pAvbsfContext, &avPacket);
            av_bsf_receive_packet(pAvbsfContext, &avPacketOut);

            if (avPacketOut.data && avPacketOut.size)
            {
                elementaryBytes.insert(elementaryBytes.end(), avPacketOut.data, avPacketOut.data + avPacketOut.size);
            }
        }
        else if (avPacket.data && avPacket.size)
        {
            elementaryBytes.insert(elementaryBytes.end(), avPacket.data, avPacket.data + avPacket.size);
        }
    };

    int  ret    = 0;
    bool isDone = false, gotVideo = false;

    interrupt_metadata.timeout_after_ms = LIBAVFORMAT_INTERRUPT_READ_TIMEOUT_MS;
    get_monotonic_time(&interrupt_metadata.value);
    fmtc->interrupt_callback.callback = _opencv_ffmpeg_interrupt_callback;
    fmtc->interrupt_callback.opaque   = &interrupt_metadata;

    while (!isDone)
    {
        ret      = av_read_frame(fmtc, &pktSrc);
        gotVideo = (pktSrc.stream_index == videoStream);
        isDone   = (ret < 0) || gotVideo;

        if (pSEIBytes && ppSEI)
        {
            // Bitstream filter lazy init;
            // We don't do this in constructor as user may not be needing SEI
            // extraction at all;
            if (!bsfc_sei)
            {
                
                // SEI has NAL type 6 for H.264 and NAL type 39 & 40 for H.265;
                const string sei_filter = is_mp4H264   ? "filter_units=pass_types=6"
                                          : is_mp4HEVC ? "filter_units=pass_types=39-40"
                                                       : "unknown";
                ret                     = av_bsf_list_parse_str(sei_filter.c_str(), &bsfc_sei);
                if (0 > ret)
                {
                    throw runtime_error("Error initializing " + sei_filter +
                                        " bitstream filter: " + AvErrorToString(ret));
                }

                ret = avcodec_parameters_copy(bsfc_sei->par_in, fmtc->streams[videoStream]->codecpar);
                if (0 != ret)
                {
                    throw runtime_error("Error copying codec parameters: " + AvErrorToString(ret));
                }

                ret = av_bsf_init(bsfc_sei);
                if (0 != ret)
                {
                    throw runtime_error("Error initializing " + sei_filter +
                                        " bitstream filter: " + AvErrorToString(ret));
                }
            }

            // Extract SEI NAL units from packet;
            auto pCopyPacket = av_packet_clone(&pktSrc);
            appendBytes(seiBytes, *pCopyPacket, pktSei, bsfc_sei, videoStream, true);
            av_packet_free(&pCopyPacket);
        }

        /* Unref non-desired packets as we don't support them yet;
         */
        if (pktSrc.stream_index != videoStream)
        {
            av_packet_unref(&pktSrc);
            continue;
        }
    }

    if (ret < 0)
    {
        if (AVERROR_EOF == ret)
        {
            is_EOF = true;
        }
        else if ((fmtc) && (fmtc->pb) && (fmtc->pb->eof_reached == true))
        {
            is_EOF = true;
            cerr << "Failed to read frame: " << AvErrorToString(ret) << endl;
        }
        else if (ret == AVERROR(ETIMEDOUT)           // -110: TCP/RTSP connection timeout
                 || ret == AVERROR_EXIT)             // interrupt callback fired (open/read timeout)
        {
            is_ConnectTimeout = true;
            cerr << "Failed to read frame (timeout): " << AvErrorToString(ret) << endl;
        }
        else
        {
            // Other errors (e.g. UDP packet loss, I/O error): log but do not mark
            // as timeout or EOF so SendBitStream can retry via read_times counter.
            cerr << "Failed to read frame: " << AvErrorToString(ret) << " (" << ret << ")" << endl;
        }
        return false;
    }

    // if (ret < 0) {
    //   if (AVERROR_EOF != ret) {
    //     // No need to report EOF;
    //     cerr << "Failed to read frame: " << AvErrorToString(ret) << endl;
    //   }
    //   return false;
    // }

    const bool bsf_needed = is_mp4H264 || is_mp4HEVC || is_AVS2;
    appendBytes(annexbBytes, pktSrc, pktDst, bsfc_annexb, videoStream, bsf_needed);

    if (gotVideo)
    {
        if (!addHeader)
        {
            if (!is_rtsp)
                ifType = 0;
            else {
                ifType = check_i_frame((char *)annexbBytes.data(), annexbBytes.size(), eVideoCodec);
            }

            if (ifType >= 0)
            {
                if (!is_AVS2) {
                    if ((ifType == 0x5) || ((ifType >= 0x10) && (ifType <= 0x15)))
                    {
                        vector<uint8_t> extradata;
                        int extradata_size = bsfc_annexb->par_in->extradata_size;
                        extradata.resize(extradata_size);
                        memcpy(extradata.data(), bsfc_annexb->par_in->extradata, extradata_size);
                        annexbBytes.insert(annexbBytes.begin(), extradata.begin(), extradata.end());
                    }
                    else if ((ifType == 0x7) || (ifType == 0x20) || (ifType == 0))
                    {
                        ;
                    }
                    else
                    {
                        cerr << "Failed to read Header frame: ifType=" << ifType << endl;
                        return false;
                    }
                } else {
                    if (ifType == 1 || ifType == 2) {
                        
                    } else {
                        // cerr << "Failed to read AVS2 header frame: ifType=" << ifType << endl;
                        return false;
                    }
                }
                addHeader = true;
            }
            else
            {
                cerr << "Failed to read Header frame: ifType=" << ifType << endl;
                return false;
            }
        }


    }

    pVideo      = annexbBytes.data();
    rVideoBytes = annexbBytes.size();

    /* Save packet props to PacketData, decoder will use it later.
     * If no BSF filters were applied, copy input packet props.
     */
    if (!bsf_needed)
    {
        av_packet_copy_props(&pktDst, &pktSrc);
    }

    last_packet_data.key      = pktDst.flags & AV_PKT_FLAG_KEY;
    last_packet_data.pts      = pktDst.pts;
    last_packet_data.dts      = pktDst.dts;
    last_packet_data.pos      = pktDst.pos;
    last_packet_data.duration = pktDst.duration;

    pktData = last_packet_data;

    if (pSEIBytes && ppSEI && !seiBytes.empty())
    {
        *ppSEI     = seiBytes.data();
        *pSEIBytes = seiBytes.size();
    }

    return true;
}

void FFmpegDemuxer::Flush()
{
    avio_flush(fmtc->pb);
    avformat_flush(fmtc);
}

int64_t FFmpegDemuxer::TsFromTime(double ts_sec)
{
    /* Internal timestamp representation is integer, so multiply to AV_TIME_BASE
     * and switch to fixed point precision arithmetics; */
    auto const ts_tbu = llround(ts_sec * AV_TIME_BASE);

    // Rescale the timestamp to value represented in stream base units;
    AVRational factor;
    factor.num = 1;
    factor.den = AV_TIME_BASE;
    return av_rescale_q(ts_tbu, factor, fmtc->streams[videoStream]->time_base);
}

int64_t FFmpegDemuxer::TsFromFrameNumber(int64_t frame_num)
{
    auto const ts_sec = (double)frame_num / GetFramerate();
    return TsFromTime(ts_sec);
}

bool FFmpegDemuxer::Seek(SeekContext& seekCtx,
                         uint8_t*&    pVideo,
                         size_t&      rVideoBytes,
                         PacketData&  pktData,
                         uint8_t**    ppSEI,
                         size_t*      pSEIBytes)
{
    /* !!! IMPORTANT !!!
     * Across this function packet decode timestamp (DTS) values are used to
     * compare given timestamp against. This is done for reason. DTS values shall
     * monotonically increase during the course of decoding unlike PTS velues
     * which may be affected by frame reordering due to B frames presence.
     */

    if (!is_seekable)
    {
        cerr << "Seek isn't supported for this input." << endl;
        return false;
    }

    if (IsVFR() && seekCtx.IsByNumber())
    {
        cerr << "Can't seek by frame number in VFR sequences. Seek by timestamp "
                "instead."
             << endl;
        return false;
    }

    // Seek for single frame;
    auto seek_frame = [&](SeekContext const& seek_ctx, int flags)
    {
        bool    seek_backward = false;
        int64_t timestamp     = 0;
        int     ret           = 0;

        if (seek_ctx.IsByNumber())
        {
            timestamp     = TsFromFrameNumber(seek_ctx.seek_frame);
            seek_backward = last_packet_data.dts > timestamp;
            ret           = av_seek_frame(
                fmtc, GetVideoStreamIndex(), timestamp, seek_backward ? AVSEEK_FLAG_BACKWARD | flags : flags);
        }
        else if (seek_ctx.IsByTimestamp())
        {
            timestamp     = TsFromTime(seek_ctx.seek_tssec);
            seek_backward = last_packet_data.dts > timestamp;
            ret           = av_seek_frame(
                fmtc, GetVideoStreamIndex(), timestamp, seek_backward ? AVSEEK_FLAG_BACKWARD | flags : flags);
        }
        else
        {
            throw runtime_error("Invalid seek mode");
        }

        if (ret < 0)
        {
            throw runtime_error("Error seeking for frame: " + AvErrorToString(ret));
        }
    };

    // Check if frame satisfies seek conditions;
    auto is_seek_done = [&](PacketData& pkt_data, SeekContext const& seek_ctx)
    {
        int64_t target_ts = 0;

        if (seek_ctx.IsByNumber())
        {
            target_ts = TsFromFrameNumber(seek_ctx.seek_frame);
        }
        else if (seek_ctx.IsByTimestamp())
        {
            // Rely solely on FFMpeg API for seek by timestamp;
            return 1;
        }
        else
        {
            throw runtime_error("Invalid seek mode.");
        }

        if (pkt_data.dts == target_ts)
        {
            return 0;
        }
        else if (pkt_data.dts > target_ts)
        {
            return 1;
        }
        else
        {
            return -1;
        };
    };

    /* This will seek for exact frame number;
     * Note that decoder may not be able to decode such frame; */
    auto seek_for_exact_frame = [&](PacketData& pkt_data, SeekContext& seek_ctx)
    {
        // Repetititive seek until seek condition is satisfied;
        SeekContext tmp_ctx = seek_ctx;
        seek_frame(tmp_ctx, AVSEEK_FLAG_ANY);

        int condition = 0;
        do
        {
            if (!Demux(pVideo, rVideoBytes, pkt_data, ppSEI, pSEIBytes))
            {
                break;
            }
            condition = is_seek_done(pkt_data, seek_ctx);

            // We've gone too far and need to seek backwards;
            if (condition > 0)
            {
                if (tmp_ctx.IsByNumber())
                {
                    tmp_ctx.seek_frame--;
                }
                else if (tmp_ctx.IsByTimestamp())
                {
                    tmp_ctx.seek_tssec -= this->GetTimebase();
                    tmp_ctx.seek_tssec = max(0.0, tmp_ctx.seek_tssec);
                }
                else
                {
                    throw runtime_error("Invalid seek mode.");
                }
                seek_frame(tmp_ctx, AVSEEK_FLAG_ANY);
            }
            // Need to read more frames until we reach requested number;
            else if (condition < 0)
            {
                continue;
            }
        } while (0 != condition);

        seek_ctx.out_frame_pts      = pkt_data.pts;
        seek_ctx.out_frame_duration = pkt_data.duration;
    };

    // Seek for closest key frame in the past;
    auto seek_for_prev_key_frame = [&](PacketData& pkt_data, SeekContext& seek_ctx)
    {
        seek_frame(seek_ctx, AVSEEK_FLAG_BACKWARD);

        Demux(pVideo, rVideoBytes, pkt_data, ppSEI, pSEIBytes);
        seek_ctx.out_frame_pts      = pkt_data.pts;
        seek_ctx.out_frame_duration = pkt_data.duration;
    };

    switch (seekCtx.mode)
    {
        case EXACT_FRAME: seek_for_exact_frame(pktData, seekCtx); break;
        case PREV_KEY_FRAME: seek_for_prev_key_frame(pktData, seekCtx); break;
        default: throw runtime_error("Unsupported seek mode"); break;
    }

    return true;
}

int FFmpegDemuxer::ReadPacket(void* opaque, uint8_t* pBuf, int nBuf)
{
    return 0;
}

AVCodecID FFmpegDemuxer::GetVideoCodec() const
{
    return eVideoCodec;
}

FFmpegDemuxer::~FFmpegDemuxer()
{
    if (pktSrc.data)
    {
        av_packet_unref(&pktSrc);
    }
    if (pktDst.data)
    {
        av_packet_unref(&pktDst);
    }

    if (bsfc_annexb)
    {
        av_bsf_free(&bsfc_annexb);
    }

    if (bsfc_sei)
    {
        av_bsf_free(&bsfc_sei);
    }

    avformat_close_input(&fmtc);

    if (avioc)
    {
        av_freep(&avioc->buffer);
        av_freep(&avioc);
    }
}

AVFormatContext* FFmpegDemuxer::CreateFormatContext(DataProvider&              pDataProvider,
                                                    const map<string, string>& ffmpeg_options)
{
    AVFormatContext* ctx = avformat_alloc_context();
    if (!ctx)
    {
        cerr << "Can't allocate AVFormatContext at " << __FILE__ << " " << __LINE__;
        return nullptr;
    }

    interrupt_metadata.timeout_after_ms = LIBAVFORMAT_INTERRUPT_OPEN_TIMEOUT_MS;
    get_monotonic_time(&interrupt_metadata.value);

    ctx->interrupt_callback.callback = _opencv_ffmpeg_interrupt_callback;
    ctx->interrupt_callback.opaque   = &interrupt_metadata;

    uint8_t* avioc_buffer      = nullptr;
    int      avioc_buffer_size = 8 * 1024 * 1024;
    avioc_buffer               = (uint8_t*)av_malloc(avioc_buffer_size);
    if (!avioc_buffer)
    {
        cerr << "Can't allocate avioc_buffer at " << __FILE__ << " " << __LINE__;
        return nullptr;
    }
    avioc = avio_alloc_context(avioc_buffer, avioc_buffer_size, 0, &pDataProvider, &ReadPacket, nullptr, nullptr);

    if (!avioc)
    {
        cerr << "Can't allocate AVIOContext at " << __FILE__ << " " << __LINE__;
        return nullptr;
    }
    ctx->pb = avioc;

    // Set up format context options;
    AVDictionary* options = NULL;
    for (auto& pair : ffmpeg_options)
    {
        auto err = av_dict_set(&options, pair.first.c_str(), pair.second.c_str(), 0);
        if (err < 0)
        {
            cerr << "Can't set up dictionary option: " << pair.first << " " << pair.second << ": "
                 << AvErrorToString(err) << "\n";
            av_dict_free(&options);
            return nullptr;
        }
    }

    auto err = avformat_open_input(&ctx, nullptr, nullptr, &options);
    if (0 != err)
    {
        cerr << "Can't open input. Error message: " << AvErrorToString(err);
        av_dict_free(&options);
        return nullptr;
    }
    av_dict_free(&options);
    return ctx;
}

AVFormatContext* FFmpegDemuxer::CreateFormatContext(const char* szFilePath, const map<string, string>& ffmpeg_options)
{
    avformat_network_init();

    // Set up format context options;
    AVDictionary* options = NULL;

    for (auto& pair : ffmpeg_options)
    {
        cout << pair.first << ": " << pair.second << endl;
        auto err = av_dict_set(&options, pair.first.c_str(), pair.second.c_str(), 0);
        if (err < 0)
        {
            cerr << "Can't set up dictionary option: " << pair.first << " " << pair.second << ": "
                 << AvErrorToString(err) << "\n";
            av_dict_free(&options);
            return nullptr;
        }
    }

    AVFormatContext* ctx = nullptr;
    // av_register_all();

    auto err = avformat_open_input(&ctx, szFilePath, nullptr, &options);
    if (err < 0 || nullptr == ctx)
    {
        cerr << "Can't open " << szFilePath << ": " << AvErrorToString(err) << "\n";
        av_dict_free(&options);
        return nullptr;
    }

    interrupt_metadata.timeout_after_ms = LIBAVFORMAT_INTERRUPT_OPEN_TIMEOUT_MS;
    get_monotonic_time(&interrupt_metadata.value);

    ctx->interrupt_callback.callback = _opencv_ffmpeg_interrupt_callback;
    ctx->interrupt_callback.opaque   = &interrupt_metadata;

    // ctx->flags |= AVFMT_FLAG_NOBUFFER;

    av_dict_free(&options);
    return ctx;
}

FFmpegDemuxer::FFmpegDemuxer(AVFormatContext* fmtcx)
    : fmtc(fmtcx)
{
    pktSrc = {};
    pktDst = {};

    memset(&last_packet_data, 0, sizeof(last_packet_data));

    if (!fmtc)
    {
        stringstream ss;
        ss << __FUNCTION__ << ": no AVFormatContext provided." << endl;
        throw invalid_argument(ss.str());
    }

    auto ret = avformat_find_stream_info(fmtc, nullptr);
    if (0 != ret)
    {
        stringstream ss;
        ss << __FUNCTION__ << ": can't find stream info;" << AvErrorToString(ret) << endl;
        throw runtime_error(ss.str());
    }

    videoStream = av_find_best_stream(fmtc, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (videoStream < 0)
    {
        stringstream ss;
        ss << __FUNCTION__ << ": can't find video stream in input file." << endl;
        throw runtime_error(ss.str());
    }

    // gop_size = fmtc->streams[videoStream]->codec->gop_size;
    eVideoCodec = fmtc->streams[videoStream]->codecpar->codec_id;
    width       = fmtc->streams[videoStream]->codecpar->width;
    height      = fmtc->streams[videoStream]->codecpar->height;
    framerate =
        (double)fmtc->streams[videoStream]->r_frame_rate.num / (double)fmtc->streams[videoStream]->r_frame_rate.den;
    avg_framerate =
        (double)fmtc->streams[videoStream]->avg_frame_rate.num / (double)fmtc->streams[videoStream]->avg_frame_rate.den;
    timebase = (double)fmtc->streams[videoStream]->time_base.num / (double)fmtc->streams[videoStream]->time_base.den;
    eChromaFormat = (AVPixelFormat)fmtc->streams[videoStream]->codecpar->format;
    nb_frames     = fmtc->streams[videoStream]->nb_frames;
    color_space   = fmtc->streams[videoStream]->codecpar->color_space;
    color_range   = fmtc->streams[videoStream]->codecpar->color_range;

    is_mp4H264 = (eVideoCodec == AV_CODEC_ID_H264);
    is_mp4HEVC = (eVideoCodec == AV_CODEC_ID_HEVC);
    is_VP9     = (eVideoCodec == AV_CODEC_ID_VP9);
    is_AVS2     = (eVideoCodec == AV_CODEC_ID_AVS2);
    av_init_packet(&pktSrc);
    pktSrc.data = nullptr;
    pktSrc.size = 0;
    av_init_packet(&pktDst);
    pktDst.data = nullptr;
    pktDst.size = 0;
    av_init_packet(&pktSei);
    pktSei.data = nullptr;
    pktSei.size = 0;

    // Initialize Annex.B BSF;
    const string bfs_name = is_mp4H264   ? "h264_mp4toannexb"
                        : is_mp4HEVC ? "hevc_mp4toannexb"
                        : is_AVS2    ? "dump_extra"
                        : is_VP9     ? string()
                                     : "unknown";


    if (!bfs_name.empty())
    {
        const AVBitStreamFilter* toAnnexB = av_bsf_get_by_name(bfs_name.c_str());
        if (!toAnnexB)
        {
            throw runtime_error("can't get " + bfs_name + " filter by name");
        }
        ret = av_bsf_alloc(toAnnexB, &bsfc_annexb);
        if (0 != ret)
        {
            throw runtime_error("Error allocating " + bfs_name + " filter: " + AvErrorToString(ret));
        }

        ret = avcodec_parameters_copy(bsfc_annexb->par_in, fmtc->streams[videoStream]->codecpar);
        if (0 != ret)
        {
            throw runtime_error("Error copying codec parameters: " + AvErrorToString(ret));
        }

        ret = av_bsf_init(bsfc_annexb);
        if (0 != ret)
        {
            throw runtime_error("Error initializing " + bfs_name + " bitstream filter: " + AvErrorToString(ret));
        }
    }

    // SEI extraction filter has lazy init as this feature is optional;
    bsfc_sei = nullptr;

    /* Some inputs doesn't allow seek functionality.
     * Check this ahead of time. */
    is_seekable = fmtc->iformat->read_seek || fmtc->iformat->read_seek2;
}
