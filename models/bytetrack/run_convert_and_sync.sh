#!/usr/bin/env bash
#
# Usage:
#   bash run_convert_and_sync.sh              # default: MAX_BATCH=4
#   MAX_BATCH=8 bash run_convert_and_sync.sh  # custom batch size
#
# Environment variables forwarded to the convert script:
#   MAX_BATCH       - maximum batch size compiled into the engine (default 4)
#   IXRT_PLUGIN_SO  - path to liboss_ixrt_plugin.so

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BYTETRACK_HOME="/home/qihang.zhang/code/model_infer/ByteTrack_bk"
CONVERT_SCRIPT="${BYTETRACK_HOME}/convert/run_convert_and_verify_ixrt.sh"

MAX_BATCH="${MAX_BATCH:-4}"
PLUGIN_PATH="${IXRT_PLUGIN_SO:-/home/qihang.zhang/code/sw_code/ixrt/third_party/ixrt-oss/build_plugin/lib/liboss_ixrt_plugin.so}"

SRC_ONNX="${BYTETRACK_HOME}/convert/bytetrack_s_with_decoder_nms.onnx"
SRC_ENGINE="${BYTETRACK_HOME}/convert/ixrt_out/bytetrack_s_with_decoder_nms_b${MAX_BATCH}.engine"

DST_ONNX="${SCRIPT_DIR}/bytetrack_s_with_decoder_nms.onnx"
DST_ENGINE="${SCRIPT_DIR}/bytetrack_s_with_decoder_nms_b${MAX_BATCH}.engine"

if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
    echo "convert script not found: ${CONVERT_SCRIPT}" >&2
    exit 1
fi

if [[ ! -f "${PLUGIN_PATH}" ]]; then
    echo "plugin not found: ${PLUGIN_PATH}" >&2
    echo "please set IXRT_PLUGIN_SO to the rebuilt liboss_ixrt_plugin.so" >&2
    exit 1
fi

echo "[1/3] run ByteTrack convert + verify  (MAX_BATCH=${MAX_BATCH})"
MAX_BATCH="${MAX_BATCH}" IXRT_PLUGIN_SO="${PLUGIN_PATH}" bash "${CONVERT_SCRIPT}"

if [[ ! -f "${SRC_ONNX}" ]]; then
    echo "converted onnx not found: ${SRC_ONNX}" >&2
    exit 1
fi

if [[ ! -f "${SRC_ENGINE}" ]]; then
    echo "converted engine not found: ${SRC_ENGINE}" >&2
    exit 1
fi

echo "[2/3] sync artifacts into iluvatarpipeline/models/bytetrack"
cp "${SRC_ONNX}"   "${DST_ONNX}"
cp "${SRC_ENGINE}" "${DST_ENGINE}"

echo "[3/3] done"
echo "onnx   -> ${DST_ONNX}"
echo "engine -> ${DST_ENGINE}  (batch 1..${MAX_BATCH})"
echo
echo "Update config/pipeline_bytetrack.json:"
echo "  \"engine_file\": \"../models/bytetrack/bytetrack_s_with_decoder_nms_b${MAX_BATCH}.engine\""
echo "  \"pre_maxBatch\": ${MAX_BATCH}"
echo "  \"infer_maxBatch\": ${MAX_BATCH}"
echo
