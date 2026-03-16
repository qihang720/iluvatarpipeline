#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"
using json = nlohmann::json;

struct ModelParams
{
    int                      resize_h       = 0;
    int                      resize_w       = 0;
    int                      pre_maxBatch   = 0;
    int                      pre_qsz        = 0;
    int                      infer_maxBatch = 0;
    int                      infer_qsz      = 0;
    std::vector<std::string> custom_outputs;
    std::string              onnx_file;
    std::string              engine_file;
};

struct RtspUrlParams
{
    std::string rtsp_url;
    // Controls inference (tracking) frequency for pipeline_bytetrack.
    // Semantics differ from other pipelines:
    //   <=1 : run inference on every decoded frame.
    //    N>1: run inference on 1 out of every N frames (skip factor, not target FPS).
    //         e.g. rate=2 → track every other frame; rate=3 → track 1 in 3, etc.
    // The video encoder always uses the full decoded stream regardless of this value.
    //
    // Other pipelines (pipeline_one_stage, pipeline_two_stage) still interpret
    // rate as a target FPS passed directly to the hardware decoder.
    int rate = -1;
};

struct GpuRtspGroup
{
    int device_id;
    std::vector<RtspUrlParams> rtsp_params;
};

namespace nlohmann
{
template <>
struct adl_serializer<ModelParams>
{
    static void from_json(const json& j, ModelParams& params)
    {
        j.at("resize_h").get_to(params.resize_h);
        j.at("resize_w").get_to(params.resize_w);
        j.at("pre_maxBatch").get_to(params.pre_maxBatch);
        j.at("pre_qsz").get_to(params.pre_qsz);
        j.at("infer_maxBatch").get_to(params.infer_maxBatch);
        j.at("infer_qsz").get_to(params.infer_qsz);
        j.at("custom_outputs").get_to(params.custom_outputs);
        j.at("onnx_file").get_to(params.onnx_file);
        j.at("engine_file").get_to(params.engine_file);
    }
};

template <>
struct adl_serializer<RtspUrlParams>
{
    static void to_json(json& j, const RtspUrlParams& params)
    {
        j = json{{"rtsp_url", params.rtsp_url}, {"rate", params.rate}};
    }

    static void from_json(const json& j, RtspUrlParams& params)
    {
        j.at("rtsp_url").get_to(params.rtsp_url);
        j.at("rate").get_to(params.rate);
    }
};

template <>
struct adl_serializer<GpuRtspGroup>
{
    static void from_json(const json& j, GpuRtspGroup& group)
    {
        j.at("device_id").get_to(group.device_id);
        j.at("rtsp_params").get_to(group.rtsp_params);
    }

    static void to_json(json& j, const GpuRtspGroup& group)
    {
        j = json{{"device_id", group.device_id}, {"rtsp_params", group.rtsp_params}};
    }
};
}  // namespace nlohmann

void from_json(const json& j, std::vector<std::string>& vec);

class RtspUrlManager
{
public:
    RtspUrlManager(const std::string& file_path) { loadFromFile(file_path); }

    const std::vector<GpuRtspGroup>& getGpuGroups() const { return gpu_groups; }

private:
    std::vector<GpuRtspGroup> gpu_groups;

    void loadFromFile(const std::string& file_path)
    {
        std::ifstream file(file_path);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open config file: " + file_path);
        }

        json j;
        file >> j;
        j.at("gpu_devices").get_to(gpu_groups);
    }
};