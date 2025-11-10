#include "pipeline_two_stage.h"

using json = nlohmann::json;

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <device id>"
                  << " <rtsp files>" << std::endl;
        return 1;
    }
    initializeLogger();
    RtspUrlManager                   rtsp_manager(argv[2]);
    const std::vector<GpuRtspGroup>  gpu_groups   = rtsp_manager.getGpuGroups();
    if (gpu_groups.empty() || gpu_groups.size() > 1){
        logger->error("No GPU groups found in the RTSP configuration or multiple GPU groups found. This test only supports a single GPU group.");
        return 1;
    }

    if (gpu_groups[0].device_id != std::stoi(argv[1])){
        logger->error("The device ID provided does not match the device ID in the RTSP configuration.");
        return 1;
    }
    const std::vector<RtspUrlParams> rtsp_sources = gpu_groups[0].rtsp_params;

    int       dev       = std::stoi(argv[1]);
    CUcontext cuContext = Init(dev);

    std::ifstream pipe_file("../config/pipeline_two_stage.json");
    json          pipe_json;
    pipe_file >> pipe_json;

    int jpeg_maxBatch = pipe_json["jpeg_maxBatch"].get<int>();
    int jpeg_qsz      = pipe_json["jpeg_qsz"].get<int>();
    int dec_qsz       = pipe_json["dec_qsz"].get<int>();

    
    
    // PPYoloE parameters
    ModelParams ppyoloe_params = pipe_json["ppyoloe_params"].get<ModelParams>();

    // PPLCNet parameters
    ModelParams pplcnet_params = pipe_json["pplcnet_params"].get<ModelParams>();

    // Generate engine file path if not provided
    if (ppyoloe_params.engine_file.empty())
    {
        ppyoloe_params.engine_file = generateEnginePath(ppyoloe_params.onnx_file);
    }

    if (pplcnet_params.engine_file.empty())
    {
        pplcnet_params.engine_file = generateEnginePath(pplcnet_params.onnx_file);
    }

    PipeLineProcessorTwoStage* PipeLine = new PipeLineProcessorTwoStage(
        cuContext, dev, jpeg_maxBatch, jpeg_qsz, dec_qsz, rtsp_sources, ppyoloe_params, pplcnet_params);

    PipeLine->StartPipeline();
    int                                video_source_numbers = PipeLine->CollectVideoProcessorNumbers();
    std::vector<DecoderProcessorStats> last_video_processor_stats(video_source_numbers);
    for (size_t i = 0; i < last_video_processor_stats.size(); i++)
    {
        last_video_processor_stats[i].id              = i;
        last_video_processor_stats[i].mapFramesOK     = 0;
        last_video_processor_stats[i].receiveFramesOK = 0;
    }

    size_t last_dec = 0, last_jpeg = 0, last_ppyoloe_pre = 0, last_ppyoloe_infer = 0, last_pplcnet_pre = 0,
           last_pplcnet_infer = 0;
    size_t times              = 0;
    auto   start_time         = std::chrono::high_resolution_clock::now();
    size_t time_interval      = 10;
    while (1)
    {
        std::this_thread::sleep_for(std::chrono::seconds(time_interval));
        std::map<int, int> video_reset_vec = PipeLine->ChcekDecoderStautsAndRestart();

        size_t dec_number  = PipeLine->GetDecFrameNumber();
        size_t jpeg_number = PipeLine->GetJpegFrameNumber();

        size_t pre_ppyoloe_number   = PipeLine->GetPPYoloEPreFrameNumber();
        size_t infer_ppyoloe_number = PipeLine->GetPPYoloEInferFrameNumber();

        size_t pre_pplcnet_number   = PipeLine->GetPPLCNetPreFrameNumber();
        size_t infer_pplcnet_number = PipeLine->GetPPLCNetInferFrameNumber();

        auto   end_time   = std::chrono::high_resolution_clock::now();
        auto   duration   = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double total_time = duration.count() / 1000000;

        size_t dec_10s           = dec_number - last_dec;
        size_t jpeg_10s          = jpeg_number - last_jpeg;
        size_t pre_ppyoloe_10s   = pre_ppyoloe_number - last_ppyoloe_pre;
        size_t infer_ppyoloe_10s = infer_ppyoloe_number - last_ppyoloe_infer;

        size_t pre_pplcnet_10s   = pre_pplcnet_number - last_pplcnet_pre;
        size_t infer_pplcnet_10s = infer_pplcnet_number - last_pplcnet_infer;

        double fps_dec           = (float)dec_10s / (float)time_interval;
        double fps_jpeg          = (float)jpeg_10s / (float)time_interval;
        double fps_ppyoloe_pre   = (float)pre_ppyoloe_10s / (float)time_interval;
        double fps_ppyoloe_infer = (float)infer_ppyoloe_10s / (float)time_interval;

        double fps_pplcnet_pre   = (float)pre_pplcnet_10s / (float)time_interval;
        double fps_pplcnet_infer = (float)infer_pplcnet_10s / (float)time_interval;

        double fps_dec_t           = dec_number / total_time;
        double fps_jpeg_t          = jpeg_number / total_time;
        double fps_ppyoloe_pre_t   = pre_ppyoloe_number / total_time;
        double fps_ppyoloe_infer_t = infer_ppyoloe_number / total_time;

        double fps_pplcnet_pre_t   = pre_pplcnet_number / total_time;
        double fps_pplcnet_infer_t = infer_pplcnet_number / total_time;

        last_dec           = dec_number;
        last_jpeg          = jpeg_number;
        last_ppyoloe_pre   = pre_ppyoloe_number;
        last_ppyoloe_infer = infer_ppyoloe_number;

        last_pplcnet_pre   = pre_pplcnet_number;
        last_pplcnet_infer = infer_pplcnet_number;

        size_t dec_qsz_now  = PipeLine->GetDecQWaitNum();
        size_t dec_qsz_drop = PipeLine->GetDecQDropNum();

        size_t jpeg_qsz_now  = PipeLine->GetJpegQWaitNum();
        size_t jpeg_qsz_drop = PipeLine->GetJpegQDropNum();

        size_t pre_ppyoloe_qsz_now  = PipeLine->GetPPYoloEPreQWaitNum();
        size_t pre_ppyoloe_qsz_drop = PipeLine->GetPPYoloEPreQDropNum();

        size_t res_ppyoloe_qsz_now  = PipeLine->GetPPYoloEResQWaitNum();
        size_t res_ppyoloe_qsz_drop = PipeLine->GetPPYoloEResQDropNum();

        size_t pre_pplcnet_qsz_now  = PipeLine->GetPPLCNetPreQWaitNum();
        size_t pre_pplcnet_qsz_drop = PipeLine->GetPPLCNetPreQDropNum();

        std::ostringstream oss;
        oss << "\n"
            << "-----------------------------------------------------------\n"
            << "     |                 |        frames   |      FPS       |\n"
            << "     |     dec         | " << std::setw(12) << dec_number            << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_dec_t              << "   |\n"
            << "     |     jpeg        | " << std::setw(12) << jpeg_number           << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_jpeg_t             << "   |\n"
            << "total|  ppyoloe-pre    | " << std::setw(12) << pre_ppyoloe_number    << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_ppyoloe_pre_t      << "   |\n"
            << "     |  ppyoloe-infer  | " << std::setw(12) << infer_ppyoloe_number  << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_ppyoloe_infer_t    << "   |\n"
            << "     |  pplcnet-pre    | " << std::setw(12) << pre_pplcnet_number    << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_pplcnet_pre_t      << "   |\n"
            << "     |  pplcnet-infer  | " << std::setw(12) << infer_pplcnet_number  << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_pplcnet_infer_t    << "   |\n\n"
            << "     |     dec         | " << std::setw(12) << dec_10s               << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_dec                << "   |\n"
            << "     |     jpeg        | " << std::setw(12) << jpeg_10s              << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_jpeg               << "   |\n"
            << "10s  |  ppyoloe-pre    | " << std::setw(12) << pre_ppyoloe_10s       << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_ppyoloe_pre        << "   |\n"
            << "     |  ppyoloe-infer  | " << std::setw(12) << infer_ppyoloe_10s     << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_ppyoloe_infer      << "   |\n"
            << "     |  pplcnet-pre    | " << std::setw(12) << pre_pplcnet_10s       << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_pplcnet_pre        << "   |\n"
            << "     |  pplcnet-infer  | " << std::setw(12) << infer_pplcnet_10s     << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_pplcnet_infer      << "   |\n\n"
            << "     |                 |        Now      |      Drops     |\n"
            << "     |     dec         | " << std::setw(12) << dec_qsz_now           << "    | " << std::setw(12) << dec_qsz_drop         << "   |\n"
            << "queue|     jpeg        | " << std::setw(12) << jpeg_qsz_now          << "    | " << std::setw(12) << jpeg_qsz_drop        << "   |\n"
            << "     |  ppyoloe-pre    | " << std::setw(12) << pre_ppyoloe_qsz_now   << "    | " << std::setw(12) << pre_ppyoloe_qsz_drop << "   |\n"
            << "     |  ppyoloe-res    | " << std::setw(12) << res_ppyoloe_qsz_now   << "    | " << std::setw(12) << res_ppyoloe_qsz_drop << "   |\n"
            << "     |  pplcnet-pre    | " << std::setw(12) << pre_pplcnet_qsz_now   << "    | " << std::setw(12) << pre_pplcnet_qsz_drop << "   |\n"
            << "------------------------------------------------------------------------------\n";

        std::vector<DecoderProcessorStats> video_processor_stats = PipeLine->CollectVideoProcessorStats();
        oss << "  video decoder ID | DecoderSendFramesOK | DecoderMapFramesOK | RestartCount |\n";
        std::vector<DecoderProcessorStats> video_processor_stats_10s(video_source_numbers);
        for (size_t i = 0; i < video_processor_stats.size(); i++) {
            oss << std::setw(12) << video_processor_stats[i].id                  << "       |" 
                << std::setw(12) << video_processor_stats[i].receiveFramesOK     << "         |" 
                << std::setw(12) << video_processor_stats[i].mapFramesOK         << "        |" 
                << std::setw(8) << video_reset_vec[video_processor_stats[i].id]  << "      |\n";

            video_processor_stats_10s[i].id = last_video_processor_stats[i].id;
            if (video_processor_stats[i].id == last_video_processor_stats[i].id) {
                video_processor_stats_10s[i].receiveFramesOK = video_processor_stats[i].receiveFramesOK - last_video_processor_stats[i].receiveFramesOK;
                video_processor_stats_10s[i].mapFramesOK     = video_processor_stats[i].mapFramesOK     - last_video_processor_stats[i].mapFramesOK;
            }
        }
        oss << "\n";
        for (size_t i = 0; i < video_processor_stats_10s.size(); i++) {
            oss << std::setw(12) << video_processor_stats_10s[i].id              << "       |" 
                << std::setw(12) << video_processor_stats_10s[i].receiveFramesOK << "         |" 
                << std::setw(12) << video_processor_stats_10s[i].mapFramesOK     << "        |\n";
        }
        oss << "---------------------------------------------------------------\n\n\n";

        logger->info(oss.str());
        last_video_processor_stats = video_processor_stats;
        times++;

        if ((times >= 10) && (dec_10s == 0))
        {
            logger->error("****************dec frame == 0, pause*****************\n");
            pause();
        }
    }

    delete PipeLine;
    if (cuContext)
        checkCudaErrors(cuCtxDestroy(cuContext));
}