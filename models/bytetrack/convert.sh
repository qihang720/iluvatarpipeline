#!/usr/bin/env bash
# =============================================================================
# ByteTrack ONNX → IxRT Engine conversion script
#
# Input  : bytetrack_s.onnx  (place in this directory before running)
# Output : bytetrack_s_with_decoder_nms.onnx
#          bytetrack_s_with_decoder_nms_b<N>.engine
#
# Usage (inside the Iluvatar IxRT container / environment):
#   bash convert.sh                 # default MAX_BATCH=4
#   MAX_BATCH=8 bash convert.sh     # custom batch size
#
# Required tools:
#   ixrtexec       -- from Iluvatar CoreX SDK (available in the IxRT container)
#
# Environment variables (all optional – sensible defaults are tried):
#   MAX_BATCH          Maximum batch size compiled into the engine  (default: 4)
#   INPUT_H / INPUT_W  Model input resolution                       (default: 608 / 1088)
#   NUM_CLASS          Number of detection classes                  (default: 1, for MOT)
#   SCORE_THR          NMS score threshold                          (default: 0.1)
#   IOU_THR            NMS IoU threshold                            (default: 0.7)
#   IXRT_PLUGIN_SO     Path to liboss_ixrt_plugin.so               (auto-detected)
#   IXRT_OSS           Path to ixrt-oss directory                  (for Python API)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------- tuneable parameters ----------
MAX_BATCH="${MAX_BATCH:-4}"
INPUT_H="${INPUT_H:-608}"
INPUT_W="${INPUT_W:-1088}"
NUM_CLASS="${NUM_CLASS:-1}"
SCORE_THR="${SCORE_THR:-0.1}"
IOU_THR="${IOU_THR:-0.7}"

# ---------- locate liboss_ixrt_plugin.so ----------
# Search order:
#   1. IXRT_PLUGIN_SO env var (user override)
#   2. Standard Iluvatar CoreX container install path
#   3. Build path relative to IXRT_OSS env var
_find_plugin() {
    local candidates=(
        "${IXRT_PLUGIN_SO:-}"
        "/usr/local/corex/lib/liboss_ixrt_plugin.so"
        "${IXRT_OSS:+${IXRT_OSS}/build_plugin/lib/liboss_ixrt_plugin.so}"
        "${IXRT_OSS:+${IXRT_OSS}/../build_plugin/lib/liboss_ixrt_plugin.so}"
    )
    for p in "${candidates[@]}"; do
        [[ -n "${p}" && -f "${p}" ]] && echo "${p}" && return 0
    done
    return 1
}
IXRT_PLUGIN_SO="$(_find_plugin)" || {
    echo ""
    echo "Error: cannot locate liboss_ixrt_plugin.so."
    echo "  Set IXRT_PLUGIN_SO=/path/to/liboss_ixrt_plugin.so, or"
    echo "  set IXRT_OSS=/path/to/ixrt-oss (the ixrt-oss source directory)."
    exit 1
}

# ---------- locate ixrt Python package (for add_decode_nms_bytetrack_s.py) ----------
# The script needs ixrt.deploy.api. Search common locations.
_find_ixrt_pythonpath() {
    # Already importable? (e.g. installed via pip or the container sets PYTHONPATH)
    if python3 -c "import ixrt" >/dev/null 2>&1; then
        echo ""   # empty string means "nothing to prepend"
        return 0
    fi
    local plugin_dir
    plugin_dir="$(dirname "${IXRT_PLUGIN_SO}")"
    local candidates=(
        "${IXRT_OSS:-}"
        "$(cd "${plugin_dir}/../../.." 2>/dev/null && pwd)"   # ixrt-oss root relative to build
        "/usr/local/corex/lib/python"
    )
    for p in "${candidates[@]}"; do
        [[ -n "${p}" && -d "${p}" ]] || continue
        if PYTHONPATH="${p}:${PYTHONPATH:-}" python3 -c "import ixrt" >/dev/null 2>&1; then
            echo "${p}"
            return 0
        fi
    done
    return 1
}
IXRT_PY="$(_find_ixrt_pythonpath)" || {
    echo ""
    echo "Error: cannot import Python package 'ixrt' (needed by add_decode_nms_bytetrack_s.py)."
    echo "  Set IXRT_OSS=/path/to/ixrt-oss, or install the ixrt Python package."
    exit 1
}

# ---------- file paths ----------
SRC_ONNX="${SCRIPT_DIR}/bytetrack_s.onnx"
NMS_ONNX="${SCRIPT_DIR}/bytetrack_s_with_decoder_nms.onnx"
ENGINE="${SCRIPT_DIR}/bytetrack_s_with_decoder_nms_b${MAX_BATCH}.engine"

MIN_SHAPE="images:1x3x${INPUT_H}x${INPUT_W}"
OPT_SHAPE="images:${MAX_BATCH}x3x${INPUT_H}x${INPUT_W}"
MAX_SHAPE="images:${MAX_BATCH}x3x${INPUT_H}x${INPUT_W}"

# ---------- pre-flight checks ----------
echo "============================================"
echo " ByteTrack conversion  MAX_BATCH=${MAX_BATCH}"
echo "============================================"
echo "SRC_ONNX       : ${SRC_ONNX}"
echo "NMS_ONNX       : ${NMS_ONNX}"
echo "ENGINE         : ${ENGINE}"
echo "IXRT_PLUGIN_SO : ${IXRT_PLUGIN_SO}"
echo "shapes         : min=${MIN_SHAPE}  opt=${OPT_SHAPE}  max=${MAX_SHAPE}"
echo ""

if [[ ! -f "${SRC_ONNX}" ]]; then
    echo "Error: ${SRC_ONNX} not found."
    echo "  Export bytetrack_s.onnx from the official ByteTrack repo first:"
    echo "    git clone https://github.com/ifzhang/ByteTrack.git"
    echo "    cd ByteTrack && pip install -r requirements.txt"
    echo "    cp ${SCRIPT_DIR}/export_onnx_dynamic.py tools/export_onnx.py"
    echo "    python3 tools/export_onnx.py -f exps/example/mot/yolox_s_mix_det.py \\"
    echo "        -c pretrained/bytetrack_s_mot17.pth.tar \\"
    echo "        --output-name ${SRC_ONNX}"
    exit 1
fi

if ! command -v ixrtexec >/dev/null 2>&1; then
    echo "Error: ixrtexec not found. Run this script inside the IxRT container."
    exit 1
fi

# ---------- Step 1: ensure base ONNX has dynamic batch ----------
# If bytetrack_s.onnx was exported with a static batch=1, ixrtexec will refuse
# --min/opt/max_shape.  Detect this and patch on the fly.
DYNAMIC_ONNX="${SRC_ONNX}"
_is_dynamic=$(python3 - <<'PYEOF'
import sys, onnx
try:
    m = onnx.load(sys.argv[1])
    for inp in m.graph.input:
        if inp.name == "images":
            d = inp.type.tensor_type.shape.dim[0]
            print("yes" if d.dim_param else "no")
            sys.exit(0)
    print("no")
except Exception:
    print("no")
PYEOF
<<<"${SRC_ONNX}" python3 -c "
import sys, onnx
m = onnx.load('${SRC_ONNX}')
for inp in m.graph.input:
    if inp.name == 'images':
        d = inp.type.tensor_type.shape.dim[0]
        print('yes' if d.dim_param else 'no')
        break
" 2>/dev/null || echo "no")

if [[ "${_is_dynamic}" != "yes" ]]; then
    DYNAMIC_ONNX="${SCRIPT_DIR}/bytetrack_s_dynamic.onnx"
    echo "[0/2] Static batch detected — patching to dynamic batch → ${DYNAMIC_ONNX}"
    python3 "${SCRIPT_DIR}/make_onnx_dynamic_batch.py" \
        -i "${SRC_ONNX}" \
        -o "${DYNAMIC_ONNX}" \
        --input-h "${INPUT_H}" \
        --input-w "${INPUT_W}"
    echo "  OK: ${DYNAMIC_ONNX}"
    echo ""
fi

# ---------- Step 1: insert Decoder + NMS nodes ----------
echo "[1/2] Adding Decoder + NMS nodes → ${NMS_ONNX}"
PYTHONPATH="${IXRT_PY:+${IXRT_PY}:}${PYTHONPATH:-}" \
python3 "${SCRIPT_DIR}/add_decode_nms_bytetrack_s.py" \
    -i "${DYNAMIC_ONNX}" \
    -o "${NMS_ONNX}" \
    --num_class "${NUM_CLASS}" \
    --score_threshold "${SCORE_THR}" \
    --iou_threshold "${IOU_THR}" \
    --max_ir_version 9
echo "  OK: ${NMS_ONNX}"

# ---------- Step 2: build dynamic-batch IxRT engine ----------
echo ""
echo "[2/2] Building dynamic-batch engine (1..${MAX_BATCH}) → ${ENGINE}"
ixrtexec \
    --onnx "${NMS_ONNX}" \
    --save_engine "${ENGINE}" \
    --precision fp16 \
    --min_shape "${MIN_SHAPE}" \
    --opt_shape "${OPT_SHAPE}" \
    --max_shape "${MAX_SHAPE}" \
    --plugins "${IXRT_PLUGIN_SO}"
echo "  OK: ${ENGINE}"

echo ""
echo "============================================"
echo " Done!"
echo "============================================"
echo "ONNX   : ${NMS_ONNX}"
echo "Engine : ${ENGINE}  (supports batch 1..${MAX_BATCH})"
echo ""
echo "Update config/pipeline_bytetrack.json:"
echo '  "engine_file":    "../models/bytetrack/bytetrack_s_with_decoder_nms_b'"${MAX_BATCH}"'.engine"'
echo '  "pre_maxBatch":  '"${MAX_BATCH}"
echo '  "infer_maxBatch":'"${MAX_BATCH}"
