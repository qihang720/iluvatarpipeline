# ByteTrack IxRT Conversion

Convert the official ByteTrack model to a dynamic-batch IxRT engine for use with `pipeline_bytetrack`.

---

## Prerequisites

- **IxRT container** with `ixrtexec` available (Iluvatar CoreX SDK)
- **Python environment** with PyTorch, for the ONNX export step

---

## Step 1 — Export ONNX from official ByteTrack repo

Clone the official ByteTrack repository and install dependencies:

```bash
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
pip install -r requirements.txt
pip install cython
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install cython_bbox
```

Download the pretrained weights (ByteTrack-S on MOT17):

```bash
mkdir -p pretrained
# Download from the official ByteTrack release:
# https://github.com/ifzhang/ByteTrack/releases
# e.g. bytetrack_s_mot17.pth.tar → pretrained/bytetrack_s_mot17.pth.tar
```

Replace the upstream export script with the patched version (adds `dynamic_axes` for batch support):

```bash
cp /path/to/iluvatarpipeline/models/bytetrack/export_onnx_dynamic.py tools/export_onnx.py
```

Export ONNX:

```bash
python3 tools/export_onnx.py \
    -f exps/example/mot/yolox_s_mix_det.py \
    -c pretrained/bytetrack_s_mot17.pth.tar \
    --output-name /path/to/iluvatarpipeline/models/bytetrack/bytetrack_s.onnx
```

The exported ONNX has a dynamic batch dimension (`-1 × 3 × 608 × 1088`).

---

## Step 2 — Convert to IxRT engine

Place `bytetrack_s.onnx` in this directory, then run the conversion script **inside the IxRT container**:

```bash
cd /path/to/iluvatarpipeline/models/bytetrack

# Default MAX_BATCH=4
bash convert.sh

# Or specify a different batch size
MAX_BATCH=8 bash convert.sh
```

### What `convert.sh` does

| Step | Action |
|---|---|
| 1 | Inserts `YoloxDecoder_IXRT` + `NMS_IXRT` nodes into the ONNX graph via `add_decode_nms_bytetrack_s.py` |
| 2 | Calls `ixrtexec` with `--min_shape / --opt_shape / --max_shape` to build a dynamic-batch fp16 engine |

### Outputs

```
bytetrack_s_with_decoder_nms.onnx         ← ONNX with Decoder + NMS
bytetrack_s_with_decoder_nms_b4.engine    ← IxRT engine (batch 1..4)
```

---

## Step 3 — Update pipeline config

Edit `config/pipeline_bytetrack.json`:

```json
{
    "bytetrack_params": {
        "pre_maxBatch":   4,
        "infer_maxBatch": 4,
        "engine_file": "../models/bytetrack/bytetrack_s_with_decoder_nms_b4.engine"
    }
}
```

> **Constraint**: `pre_maxBatch` must equal `infer_maxBatch` and must not exceed `MAX_BATCH` used during conversion.

---

## Environment variables

All variables are optional. The script auto-detects common paths.

| Variable | Default (search order) | Description |
|---|---|---|
| `MAX_BATCH` | `4` | Max batch size compiled into the engine |
| `INPUT_H` | `608` | Input image height |
| `INPUT_W` | `1088` | Input image width |
| `NUM_CLASS` | `1` | Number of classes (1 for MOT) |
| `SCORE_THR` | `0.1` | NMS score threshold |
| `IOU_THR` | `0.7` | NMS IoU threshold |
| `IXRT_PLUGIN_SO` | `/usr/local/corex/lib/liboss_ixrt_plugin.so` | Path to `liboss_ixrt_plugin.so` |
| `IXRT_OSS` | — | Path to `ixrt-oss` source directory (for Python API) |

---

## Files in this directory

| File | Description |
|---|---|
| `convert.sh` | Main conversion script (ONNX → engine) |
| `export_onnx_dynamic.py` | Patched export script for ByteTrack (adds `dynamic_axes`) |
| `add_decode_nms_bytetrack_s.py` | Inserts IxRT Decoder + NMS nodes into ONNX graph |
| `bytetrack_s_with_decoder_nms.onnx` | Pre-built ONNX (Decoder + NMS) |
| `bytetrack_s_with_decoder_nms_b4.engine` | Pre-built engine (batch 1..4) |

---

## Notes

- `export_onnx_dynamic.py` is a drop-in replacement for ByteTrack's `tools/export_onnx.py`.
  The only changes vs. upstream are `dynamic_axes` in `torch.onnx.export` and
  `overwrite_input_shapes` in the `onnxsim` call, which preserve the dynamic batch dimension
  through simplification.
- `NMS_IXRT` batch support depends on the `liboss_ixrt_plugin.so` version. If results are
  incorrect with batch > 1, switch to a decoder-only engine and use the C++ NMS kernel in
  `bytetrack_cuda.cu`.
