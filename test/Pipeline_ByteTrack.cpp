#include "pipeline_bytetrack.h"

using json = nlohmann::json;

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <device id> <rtsp files>" << std::endl;
        return 1;
    }

    initializeLogger();
    RtspUrlManager rtsp_manager(argv[2]);
    const std::vector<GpuRtspGroup> gpu_groups = rtsp_manager.getGpuGroups();
    if (gpu_groups.empty() || gpu_groups.size() > 1)
    {
        logger->error("No GPU groups found in the RTSP configuration or multiple GPU groups found. This test only supports a single GPU group.");
        return 1;
    }

    if (gpu_groups[0].device_id != std::stoi(argv[1]))
    {
        logger->error("The device ID provided does not match the device ID in the RTSP configuration.");
        return 1;
    }
    const std::vector<RtspUrlParams> rtsp_sources = gpu_groups[0].rtsp_params;

    const int dev = std::stoi(argv[1]);
    CUcontext cuContext = Init(dev);

    std::ifstream pipe_file("../config/pipeline_bytetrack.json");
    json pipe_json;
    pipe_file >> pipe_json;

    const int jpeg_maxBatch = pipe_json["jpeg_maxBatch"].get<int>();
    const int jpeg_qsz = pipe_json["jpeg_qsz"].get<int>();
    const int dec_qsz = pipe_json["dec_qsz"].get<int>();

    ModelParams bytetrack_params = pipe_json["bytetrack_params"].get<ModelParams>();
    if (bytetrack_params.engine_file.empty())
    {
        bytetrack_params.engine_file = generateEnginePath(bytetrack_params.onnx_file);
    }

    const std::string encode_output_dir = pipe_json.value("encode_output_dir", std::string("./track_output"));

    PipeLineProcessorByteTrack* PipeLine =
        new PipeLineProcessorByteTrack(cuContext, dev, jpeg_maxBatch, jpeg_qsz, dec_qsz, rtsp_sources,
                                       bytetrack_params, 30, encode_output_dir);

    PipeLine->StartPipeline();
    const int video_source_numbers = PipeLine->CollectVideoProcessorNumbers();
    std::vector<DecoderProcessorStats> last_video_processor_stats(video_source_numbers);
    for (size_t i = 0; i < last_video_processor_stats.size(); ++i)
    {
        last_video_processor_stats[i].id = i;
        last_video_processor_stats[i].mapFramesOK = 0;
        last_video_processor_stats[i].receiveFramesOK = 0;
    }

    size_t last_dec = 0, last_jpeg = 0, last_bytetrack_pre = 0, last_bytetrack_infer = 0, last_encode = 0;
    size_t times = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    const size_t time_interval = 10;
    while (times < 10)
    {
        std::this_thread::sleep_for(std::chrono::seconds(time_interval));
        std::map<int, int> video_reset_vec = PipeLine->ChcekDecoderStautsAndRestart();

        const size_t dec_number = PipeLine->GetDecFrameNumber();
        const size_t jpeg_number = PipeLine->GetJpegFrameNumber();
        const size_t pre_bytetrack_number = PipeLine->GetByteTrackPreFrameNumber();
        const size_t infer_bytetrack_number = PipeLine->GetByteTrackInferFrameNumber();
        const size_t encode_number = PipeLine->GetEncodeFrameNumber();

        const auto end_time = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        const double total_time = duration.count() / 1000000.0;

        const size_t dec_10s = dec_number - last_dec;
        const size_t jpeg_10s = jpeg_number - last_jpeg;
        const size_t pre_bytetrack_10s = pre_bytetrack_number - last_bytetrack_pre;
        const size_t infer_bytetrack_10s = infer_bytetrack_number - last_bytetrack_infer;
        const size_t encode_10s = encode_number - last_encode;

        const double fps_dec = static_cast<double>(dec_10s) / static_cast<double>(time_interval);
        const double fps_jpeg = static_cast<double>(jpeg_10s) / static_cast<double>(time_interval);
        const double fps_bytetrack_pre = static_cast<double>(pre_bytetrack_10s) / static_cast<double>(time_interval);
        const double fps_bytetrack_infer = static_cast<double>(infer_bytetrack_10s) / static_cast<double>(time_interval);
        const double fps_encode = static_cast<double>(encode_10s) / static_cast<double>(time_interval);

        const double fps_dec_t = dec_number / total_time;
        const double fps_jpeg_t = jpeg_number / total_time;
        const double fps_bytetrack_pre_t = pre_bytetrack_number / total_time;
        const double fps_bytetrack_infer_t = infer_bytetrack_number / total_time;
        const double fps_encode_t = encode_number / total_time;

        last_dec = dec_number;
        last_jpeg = jpeg_number;
        last_bytetrack_pre = pre_bytetrack_number;
        last_bytetrack_infer = infer_bytetrack_number;
        last_encode = encode_number;

        const size_t dec_qsz_now = PipeLine->GetDecQWaitNum();
        const size_t dec_qsz_drop = PipeLine->GetDecQDropNum();
        const size_t jpeg_qsz_now = PipeLine->GetJpegQWaitNum();
        const size_t jpeg_qsz_drop = PipeLine->GetJpegQDropNum();
        const size_t pre_bytetrack_qsz_now = PipeLine->GetByteTrackPreQWaitNum();
        const size_t pre_bytetrack_qsz_drop = PipeLine->GetByteTrackPreQDropNum();
        const size_t res_bytetrack_qsz_now = PipeLine->GetByteTrackResQWaitNum();
        const size_t res_bytetrack_qsz_drop = PipeLine->GetByteTrackResQDropNum();
        const size_t encode_raw_qsz_now = PipeLine->GetEncodeRawQWaitNum();
        const size_t encode_raw_qsz_drop = PipeLine->GetEncodeRawQDropNum();

        std::ostringstream oss;
        oss << "\n"
            << "-----------------------------------------------------------\n"
            << "     |                 |        frames   |      FPS       |\n"
            << "     |     dec         | " << std::setw(12) << dec_number << "    | " << std::setw(12) << std::fixed
            << std::setprecision(2) << fps_dec_t << "   |\n"
            << "     |     jpeg        | " << std::setw(12) << jpeg_number << "    | " << std::setw(12) << std::fixed
            << std::setprecision(2) << fps_jpeg_t << "   |\n"
            << "total| bytetrack-pre   | " << std::setw(12) << pre_bytetrack_number << "    | " << std::setw(12)
            << std::fixed << std::setprecision(2) << fps_bytetrack_pre_t << "   |\n"
            << "     |  bytetrack      | " << std::setw(12) << infer_bytetrack_number << "    | " << std::setw(12)
            << std::fixed << std::setprecision(2) << fps_bytetrack_infer_t << "   |\n"
            << "     |  vid-encode     | " << std::setw(12) << encode_number << "    | " << std::setw(12)
            << std::fixed << std::setprecision(2) << fps_encode_t << "   |\n"
            << "     |     dec         | " << std::setw(12) << dec_10s << "    | " << std::setw(12) << std::fixed
            << std::setprecision(2) << fps_dec << "   |\n"
            << "     |     jpeg        | " << std::setw(12) << jpeg_10s << "    | " << std::setw(12) << std::fixed
            << std::setprecision(2) << fps_jpeg << "   |\n"
            << "10s  | bytetrack-pre   | " << std::setw(12) << pre_bytetrack_10s << "    | " << std::setw(12)
            << std::fixed << std::setprecision(2) << fps_bytetrack_pre << "   |\n"
            << "     |  bytetrack      | " << std::setw(12) << infer_bytetrack_10s << "    | " << std::setw(12)
            << std::fixed << std::setprecision(2) << fps_bytetrack_infer << "   |\n"
            << "     |  vid-encode     | " << std::setw(12) << encode_10s << "    | " << std::setw(12)
            << std::fixed << std::setprecision(2) << fps_encode << "   |\n"
            << "     |                 |        Now      |      Drops     |\n"
            << "     |     dec         | " << std::setw(12) << dec_qsz_now << "    | " << std::setw(12) << dec_qsz_drop
            << "   |\n"
            << "queue|     jpeg        | " << std::setw(12) << jpeg_qsz_now << "    | " << std::setw(12) << jpeg_qsz_drop
            << "   |\n"
            << "     | bytetrack-pre   | " << std::setw(12) << pre_bytetrack_qsz_now << "    | " << std::setw(12)
            << pre_bytetrack_qsz_drop << "   |\n"
            << "     |  bytetrack      | " << std::setw(12) << res_bytetrack_qsz_now << "    | " << std::setw(12)
            << res_bytetrack_qsz_drop << "   |\n"
            << "     |  encode-raw     | " << std::setw(12) << encode_raw_qsz_now << "    | " << std::setw(12)
            << encode_raw_qsz_drop << "   |\n"
            << "------------------------------------------------------------------------------\n";

        std::vector<DecoderProcessorStats> video_processor_stats = PipeLine->CollectVideoProcessorStats();
        oss << "  video decoder ID | DecoderSendFramesOK | DecoderMapFramesOK | RestartCount |\n";
        std::vector<DecoderProcessorStats> video_processor_stats_10s(video_source_numbers);
        for (size_t i = 0; i < video_processor_stats.size(); ++i)
        {
            oss << std::setw(12) << video_processor_stats[i].id << "       |" << std::setw(12)
                << video_processor_stats[i].receiveFramesOK << "         |" << std::setw(12)
                << video_processor_stats[i].mapFramesOK << "        |" << std::setw(8)
                << video_reset_vec[video_processor_stats[i].id] << "      |\n";

            video_processor_stats_10s[i].id = last_video_processor_stats[i].id;
            if (video_processor_stats[i].id == last_video_processor_stats[i].id)
            {
                video_processor_stats_10s[i].receiveFramesOK =
                    video_processor_stats[i].receiveFramesOK - last_video_processor_stats[i].receiveFramesOK;
                video_processor_stats_10s[i].mapFramesOK =
                    video_processor_stats[i].mapFramesOK - last_video_processor_stats[i].mapFramesOK;
            }
        }
        oss << "\n";
        for (size_t i = 0; i < video_processor_stats_10s.size(); ++i)
        {
            oss << std::setw(12) << video_processor_stats_10s[i].id << "       |" << std::setw(12)
                << video_processor_stats_10s[i].receiveFramesOK << "         |" << std::setw(12)
                << video_processor_stats_10s[i].mapFramesOK << "        |\n";
        }
        oss << "---------------------------------------------------------------\n\n\n";

        logger->info(oss.str());
        last_video_processor_stats = video_processor_stats;
        ++times;

        if ((times >= 10) && (dec_10s == 0))
        {
            logger->error("****************dec frame == 0, pause*****************\n");
        }
    }

    delete PipeLine;
    if (cuContext)
    {
        checkCudaErrors(cuCtxDestroy(cuContext));
    }
}
