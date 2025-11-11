#include "pipeline_one_stage.h"

using json = nlohmann::json;

std::vector<int> parseDeviceIds(const std::string& arg)
{
    std::vector<int> ids;
    std::stringstream ss(arg);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        ids.push_back(std::stoi(item));
    }
    return ids;
}

struct ProcessorStats{
    CUcontext                           cuContext;
    PipeLineProcessorOneStage*          PipeLine;
    std::vector<DecoderProcessorStats>  video_stats;

    size_t last_dec         = 0;
    size_t last_jpeg        = 0;
    size_t last_yolov5s_pre = 0;
    size_t last_yolov5s_infer = 0;
};

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <device_ids(comma separated)> <rtsp file>\n";
        std::cerr << "Example: " << argv[0] << " 0,1,2 gpu_rtsp_params.json\n";
        return 1;
    }
    initializeLogger();
    std::vector<int> device_ids = parseDeviceIds(argv[1]);

    RtspUrlManager                   rtsp_manager(argv[2]);
    const std::vector<GpuRtspGroup>  gpu_groups   = rtsp_manager.getGpuGroups();
    if (gpu_groups.size() < 2){
        logger->error("This test requires at least two GPU groups in the RTSP configuration.");
        return 1;
    }

    if (device_ids.size() != gpu_groups.size()){
        logger->error("The device ID provided does not match the device ID in the RTSP configuration.");
        return 1;
    }

    std::ifstream pipe_file("../config/pipeline_one_stage.json");
    json          pipe_json;
    pipe_file >> pipe_json;

    int jpeg_maxBatch = pipe_json["jpeg_maxBatch"].get<int>();
    int jpeg_qsz      = pipe_json["jpeg_qsz"].get<int>();
    int dec_qsz       = pipe_json["dec_qsz"].get<int>();
    
    // Yolov5s parameters
    ModelParams yolov5s_params = pipe_json["yolov5s_params"].get<ModelParams>();

    // Generate engine file path if not provided
    if (yolov5s_params.engine_file.empty())
    {
        yolov5s_params.engine_file = generateEnginePath(yolov5s_params.onnx_file);
    }

    std::map<int, ProcessorStats> processores_stats;
    for(size_t g = 0; g < device_ids.size(); g++)
    {
        if (device_ids[g] != gpu_groups[g].device_id){
            logger->error("The device ID provided does not match the device ID in the RTSP configuration.");
            return 1;
        }

        CUcontext cuContext = Init(device_ids[g]);
        PipeLineProcessorOneStage* PipeLine = new PipeLineProcessorOneStage(
            cuContext, device_ids[g], jpeg_maxBatch, jpeg_qsz, dec_qsz, gpu_groups[g].rtsp_params, yolov5s_params);

        PipeLine->StartPipeline();

        int                                video_source_numbers = PipeLine->CollectVideoProcessorNumbers();
        std::vector<DecoderProcessorStats> last_video_processor_stats(video_source_numbers);
        for (size_t i = 0; i < last_video_processor_stats.size(); i++)
        {
            last_video_processor_stats[i].id              = i;
            last_video_processor_stats[i].mapFramesOK     = 0;
            last_video_processor_stats[i].receiveFramesOK = 0;
        }
        
        processores_stats[device_ids[g]] = {cuContext, PipeLine, last_video_processor_stats, 0, 0, 0, 0};
    }

    size_t times              = 0;
    auto   start_time         = std::chrono::high_resolution_clock::now();
    size_t time_interval      = 10;
    while (1)
    {
        std::this_thread::sleep_for(std::chrono::seconds(time_interval));
        for (size_t g = 0; g < device_ids.size(); g++)
        {
            PipeLineProcessorOneStage* PipeLine = processores_stats[device_ids[g]].PipeLine;
            int dev_id = device_ids[g];
            std::map<int, int> video_reset_vec = PipeLine->ChcekDecoderStautsAndRestart();

            size_t dec_number  = PipeLine->GetDecFrameNumber();
            size_t jpeg_number = PipeLine->GetJpegFrameNumber();

            size_t pre_yolov5s_number   = PipeLine->GetYoloV5PreFrameNumber();
            size_t infer_yolov5s_number = PipeLine->GetYoloV5InferFrameNumber();

            auto   end_time   = std::chrono::high_resolution_clock::now();
            auto   duration   = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            double total_time = duration.count() / 1000000;

            size_t dec_10s           = dec_number           - processores_stats[device_ids[g]].last_dec;
            size_t jpeg_10s          = jpeg_number          - processores_stats[device_ids[g]].last_jpeg;
            size_t pre_yolov5s_10s   = pre_yolov5s_number   - processores_stats[device_ids[g]].last_yolov5s_pre;
            size_t infer_yolov5s_10s = infer_yolov5s_number - processores_stats[device_ids[g]].last_yolov5s_infer;

            double fps_dec           = (float)dec_10s           / (float)time_interval;
            double fps_jpeg          = (float)jpeg_10s          / (float)time_interval;
            double fps_yolov5s_pre   = (float)pre_yolov5s_10s   / (float)time_interval;
            double fps_yolov5s_infer = (float)infer_yolov5s_10s / (float)time_interval;


            double fps_dec_t           = dec_number             / total_time;
            double fps_jpeg_t          = jpeg_number            / total_time;
            double fps_yolov5s_pre_t   = pre_yolov5s_number     / total_time;
            double fps_yolov5s_infer_t = infer_yolov5s_number   / total_time;


            processores_stats[device_ids[g]].last_dec           = dec_number;
            processores_stats[device_ids[g]].last_jpeg          = jpeg_number;
            processores_stats[device_ids[g]].last_yolov5s_pre   = pre_yolov5s_number;
            processores_stats[device_ids[g]].last_yolov5s_infer = infer_yolov5s_number;


            size_t dec_qsz_now  = PipeLine->GetDecQWaitNum();
            size_t dec_qsz_drop = PipeLine->GetDecQDropNum();

            size_t jpeg_qsz_now  = PipeLine->GetJpegQWaitNum();
            size_t jpeg_qsz_drop = PipeLine->GetJpegQDropNum();

            size_t pre_yolov5s_qsz_now  = PipeLine->GetYoloV5PreQWaitNum();
            size_t pre_yolov5s_qsz_drop = PipeLine->GetYoloV5PreQDropNum();

            size_t res_yolov5s_qsz_now  = PipeLine->GetYoloV5ResQWaitNum();
            size_t res_yolov5s_qsz_drop = PipeLine->GetYoloV5ResQDropNum();


            std::ostringstream oss;
            oss << "\n"
                << "---------------------------- GPU "<< device_ids[g] <<"-------------------------\n"
                << "     |                 |        frames   |      FPS       |\n"
                << "     |     dec         | " << std::setw(12) << dec_number            << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_dec_t              << "   |\n"
                << "     |     jpeg        | " << std::setw(12) << jpeg_number           << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_jpeg_t             << "   |\n"
                << "total|  yolov5s-pre    | " << std::setw(12) << pre_yolov5s_number    << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_yolov5s_pre_t      << "   |\n"
                << "     |  yolov5s-infer  | " << std::setw(12) << infer_yolov5s_number  << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_yolov5s_infer_t    << "   |\n"
                << "     |     dec         | " << std::setw(12) << dec_10s               << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_dec                << "   |\n"
                << "     |     jpeg        | " << std::setw(12) << jpeg_10s              << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_jpeg               << "   |\n"
                << "10s  |  yolov5s-pre    | " << std::setw(12) << pre_yolov5s_10s       << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_yolov5s_pre        << "   |\n"
                << "     |  yolov5s-infer  | " << std::setw(12) << infer_yolov5s_10s     << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_yolov5s_infer      << "   |\n"
                << "     |                 |        Now      |      Drops     |\n"
                << "     |     dec         | " << std::setw(12) << dec_qsz_now           << "    | " << std::setw(12) << dec_qsz_drop         << "   |\n"
                << "queue|     jpeg        | " << std::setw(12) << jpeg_qsz_now          << "    | " << std::setw(12) << jpeg_qsz_drop        << "   |\n"
                << "     |  yolov5s-pre    | " << std::setw(12) << pre_yolov5s_qsz_now   << "    | " << std::setw(12) << pre_yolov5s_qsz_drop << "   |\n"
                << "     |  yolov5s-res    | " << std::setw(12) << res_yolov5s_qsz_now   << "    | " << std::setw(12) << res_yolov5s_qsz_drop << "   |\n"
                << "------------------------------------------------------------------------------\n";

            std::vector<DecoderProcessorStats> video_processor_stats = PipeLine->CollectVideoProcessorStats();
            int                                video_source_numbers  = PipeLine->CollectVideoProcessorNumbers();
            oss << "  video decoder ID | DecoderSendFramesOK | DecoderMapFramesOK | RestartCount |\n";
            std::vector<DecoderProcessorStats> video_processor_stats_10s(video_source_numbers);
            for (size_t i = 0; i < video_processor_stats.size(); i++) {
                oss << std::setw(12) << video_processor_stats[i].id                  << "       |" 
                    << std::setw(12) << video_processor_stats[i].receiveFramesOK     << "         |" 
                    << std::setw(12) << video_processor_stats[i].mapFramesOK         << "        |" 
                    << std::setw(8) << video_reset_vec[video_processor_stats[i].id]  << "      |\n";

                video_processor_stats_10s[i].id = processores_stats[device_ids[g]].video_stats[i].id;
                if (video_processor_stats[i].id == processores_stats[device_ids[g]].video_stats[i].id) {
                    video_processor_stats_10s[i].receiveFramesOK = video_processor_stats[i].receiveFramesOK - processores_stats[device_ids[g]].video_stats[i].receiveFramesOK;
                    video_processor_stats_10s[i].mapFramesOK     = video_processor_stats[i].mapFramesOK     - processores_stats[device_ids[g]].video_stats[i].mapFramesOK;
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
            processores_stats[device_ids[g]].video_stats = video_processor_stats;
            times++;

            if ((times >= 10) && (dec_10s == 0))
            {
                logger->error("****************dec frame == 0, pause*****************\n");
                pause();
            }
        }
    }

    for (size_t g = 0; g < device_ids.size(); g++)
    {
        delete processores_stats[device_ids[g]].PipeLine;
        if (processores_stats[device_ids[g]].cuContext)
            checkCudaErrors(cuCtxDestroy(processores_stats[device_ids[g]].cuContext));
    }
}