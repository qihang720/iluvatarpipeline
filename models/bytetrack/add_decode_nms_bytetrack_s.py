#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add IxRT YOLOX Decoder + NMS to ByteTrack bytetrack_s.onnx.

Your model: bytetrack_s.onnx
  - Input:  images [1, 3, 608, 1088]
  - Output: output [1, 13566, 6]  (raw: 6 = 4 reg + 1 obj + 1 class, num_class=1)

The graph has 3 head branches: cat_14 (stride 8), cat_15 (stride 16), cat_16 (stride 32).
We add Split -> YoloxDecoder_IXRT (x3) -> Concat -> NMS_IXRT -> final outputs.

Requires: ixrt Python package (from sw_code/ixrt, e.g. third_party/ixrt-oss).
  export PYTHONPATH=/path/to/ixrt/third_party/ixrt-oss:$PYTHONPATH

Usage (run from ByteTrack root or use absolute paths):
  python3 convert/add_decode_nms_bytetrack_s.py -i ../bytetrack_s.onnx -o ../bytetrack_s_with_decoder_nms.onnx
"""

import argparse
import sys
import os
import tempfile

import numpy as np

try:
    import onnx
    from onnx import helper as onnx_helper
except ImportError:
    onnx = None
    onnx_helper = None

try:
    from ixrt.deploy.api import GraphTransform, create_source, create_target
except ImportError as e:
    print("Error: need ixrt.", e)
    print("  export IXRT_OSS=/path/to/ixrt-oss   # or PYTHONPATH=$IXRT_OSS:$PYTHONPATH")
    print("  Example: export PYTHONPATH=/home/qihang.zhang/code/sw_code/ixrt/third_party/ixrt-oss:$PYTHONPATH")
    sys.exit(1)


# 默认 head 名称（与 export 得到的 bytetrack_s.onnx 一致；也可由 discover_head_names_from_onnx 自动检测）
HEAD_STRIDE8  = "cat_14"   # (1, 6, 76, 136)
HEAD_STRIDE16 = "cat_15"   # (1, 6, 38, 68)
HEAD_STRIDE32 = "cat_16"   # (1, 6, 19, 34)
NUM_CLASS = 1
SPLIT_CHANNELS = [4, 1, NUM_CLASS]  # reg, obj, cls

# 期望的 3 个 head 的 spatial shape (H, W)，用于从 ONNX 中按形状匹配
EXPECTED_HEAD_SHAPES = [(76, 136), (38, 68), (19, 34)]  # stride 8, 16, 32 for input 608x1088


def discover_head_names_from_onnx(onnx_path: str, output_name: str = "output"):
    """
    从 ONNX 图中追溯 output 的输入，得到三个 head 张量名（顺序：stride8, stride16, stride32）。
    返回 (head_8, head_16, head_32) 或 None（检测失败时用默认名）。
    """
    if onnx is None:
        return None
    try:
        model = onnx.load(onnx_path)
    except Exception as e:
        print("Warning: could not load ONNX for head discovery:", e)
        return None
    out_to_node = {}
    for n in model.graph.node:
        for o in n.output:
            out_to_node[o] = n
    # 找到产出 output 的节点，其输入是 cat_17
    if output_name not in out_to_node:
        print("Warning: output '%s' not found in graph" % output_name)
        return None
    node = out_to_node[output_name]
    if node.op_type != "Transpose" or len(node.input) != 1:
        print("Warning: output producer is not Transpose or has !=1 input:", node.op_type, node.input)
        return None
    cat_name = node.input[0]  # e.g. cat_17
    if cat_name not in out_to_node:
        return None
    concat_node = out_to_node[cat_name]
    if concat_node.op_type != "Concat" or len(concat_node.input) != 3:
        print("Warning: concat node has !=3 inputs:", concat_node.input)
        return None
    # concat 的 3 个输入是 view, view_1, view_2；各自由 Reshape( head_tensor ) 得到
    view_names = concat_node.input
    head_names = []
    for v in view_names:
        if v not in out_to_node:
            return None
        reshape_node = out_to_node[v]
        if reshape_node.op_type != "Reshape" or len(reshape_node.input) < 1:
            return None
        head_names.append(reshape_node.input[0])
    if len(head_names) != 3:
        return None
    print("Discovered head names from ONNX (stride 8, 16, 32):", head_names)
    return tuple(head_names)


def _fix_onnx_for_ixrt(onnx_path: str, max_ir_version: int = 9) -> str:
    """Fix ONNX so ixrt converter can load it. Optionally downgrade ir_version, fix Conv/MaxPool/Resize. Returns path to fixed ONNX (temp file if modified)."""
    if onnx is None or onnx_helper is None:
        print("Warning: 'onnx' package not installed; cannot fix ONNX (pip install onnx).")
        return onnx_path
    model = onnx.load(onnx_path)
    initializer_shapes = {}
    for init in model.graph.initializer:
        dims = getattr(init, "dims", None) or []
        initializer_shapes[init.name] = list(dims)
    modified = False
    if getattr(model, "ir_version", 0) > max_ir_version:
        model.ir_version = max_ir_version
        modified = True
        print("  Downgraded ir_version to %d for ixrt." % max_ir_version)
    for node in model.graph.node:
        # 1) Conv: add kernel_shape if missing (ixrt ConvAttr requires it)
        if node.op_type == "Conv":
            has_kernel = any(a.name == "kernel_shape" for a in node.attribute)
            if not has_kernel and len(node.input) >= 2:
                w_name = node.input[1]
                if w_name in initializer_shapes:
                    w_shape = initializer_shapes[w_name]
                    if len(w_shape) >= 4:
                        kH, kW = int(w_shape[2]), int(w_shape[3])
                        node.attribute.append(onnx_helper.make_attribute("kernel_shape", [kH, kW]))
                        modified = True
        # 2) MaxPool: remove storage_order (ixrt MaxPoolAttr does not accept it)
        if node.op_type == "MaxPool":
            to_remove = [i for i, a in enumerate(node.attribute) if a.name == "storage_order"]
            if to_remove:
                for i in reversed(to_remove):
                    del node.attribute[i]
                modified = True
        # 3) Resize: remove attrs ixrt ResizeAttr does not accept
        if node.op_type == "Resize":
            unsupported = {"keep_aspect_ratio_policy", "antialias"}
            to_remove = [i for i, a in enumerate(node.attribute) if a.name in unsupported]
            if to_remove:
                for i in reversed(to_remove):
                    del node.attribute[i]
                modified = True
    if not modified:
        return onnx_path
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp.close()
    onnx.save(model, tmp.name)
    return tmp.name


class ByteTrackYoloXTransform:
    def __init__(self, graph):
        self.t = GraphTransform(graph)
        self.graph = graph

    def AddSplitOp(self, name: str, input_name: str, output_names: list, axis: int = 1, split: list = None):
        inputs = [input_name]
        if split is not None:
            split_var = self.t.make_variable(
                name=f"{name}_split_sizes",
                value=np.array(split, dtype=np.int64),
            )
            inputs.append(split_var.name)
        self.t.make_operator("Split", name=name, inputs=inputs, outputs=output_names, axis=axis)
        return self.graph

    def AddYoloDecoderOp(self, name: str, inputs: list, outputs: list, **attributes):
        self.t.make_operator(
            "YoloxDecoder_IXRT", name=name, inputs=inputs, outputs=outputs, **attributes
        )
        return self.graph

    def AddConcatOp(self, name: str, inputs: list, outputs: list, **attributes):
        self.t.make_operator("Concat", name=name, inputs=inputs, outputs=outputs, **attributes)
        return self.graph

    def AddNMSOp(self, name: str, inputs: list, outputs: list, **attributes):
        self.t.make_operator("NMS_IXRT", name=name, inputs=inputs, outputs=outputs, **attributes)
        for var_name in outputs:
            self.t.add_output(var_name)
            self.t.get_variable(var_name).dtype = "FLOAT"
        return self.graph

    def AddOutputs(self, var_names: list, dtype: str = "FLOAT"):
        for var_name in var_names:
            self.t.add_output(var_name)
            self.t.get_variable(var_name).dtype = dtype
        return self.graph

    def DropUselessOutputs(self):
        for var_name in ["output"]:
            self.t.delete_output(var_name)
        return self.graph

    def Cleanup(self):
        self.t.cleanup()
        return self.graph


def add_decode_only(graph, num_class: int = 1,
                    head_8: str = None, head_16: str = None, head_32: str = None):
    """仅添加 Decoder（Split + YoloxDecoder x3 + Concat），输出 boxes 和 scores，不添加 NMS。用于与 demo_postprocess 对比验证 decoder 层。"""
    h8 = head_8 or HEAD_STRIDE8
    h16 = head_16 or HEAD_STRIDE16
    h32 = head_32 or HEAD_STRIDE32
    t = ByteTrackYoloXTransform(graph)
    split = SPLIT_CHANNELS if num_class == 1 else [4, 1, num_class]

    t.AddSplitOp("split_8",  h8,  ["loc_8",  "conf_8",  "cls_8"],  axis=1, split=split)
    t.AddSplitOp("split_16", h16, ["loc_16", "conf_16", "cls_16"], axis=1, split=split)
    t.AddSplitOp("split_32", h32, ["loc_32", "conf_32", "cls_32"], axis=1, split=split)

    t.AddYoloDecoderOp(
        "decoder_8", inputs=["cls_8", "loc_8", "conf_8"],
        outputs=["decoder_8_boxes", "decoder_8_scores"], num_class=num_class, stride=8
    )
    t.AddYoloDecoderOp(
        "decoder_16", inputs=["cls_16", "loc_16", "conf_16"],
        outputs=["decoder_16_boxes", "decoder_16_scores"], num_class=num_class, stride=16
    )
    t.AddYoloDecoderOp(
        "decoder_32", inputs=["cls_32", "loc_32", "conf_32"],
        outputs=["decoder_32_boxes", "decoder_32_scores"], num_class=num_class, stride=32
    )

    t.AddConcatOp(
        "concat_boxes",
        inputs=["decoder_8_boxes", "decoder_16_boxes", "decoder_32_boxes"],
        outputs=["boxes"], axis=3
    )
    t.AddConcatOp(
        "concat_scores",
        inputs=["decoder_8_scores", "decoder_16_scores", "decoder_32_scores"],
        outputs=["scores"], axis=2
    )

    t.AddOutputs(["boxes", "scores"])
    t.DropUselessOutputs()
    t.Cleanup()
    return graph


def add_decode_nms(graph, num_class: int = 1, score_threshold: float = 0.1, iou_threshold: float = 0.7,
                   head_8: str = None, head_16: str = None, head_32: str = None):
    """
    head_8, head_16, head_32: 三个 head 张量在 graph 中的名称（顺序对应 stride 8, 16, 32）。
    若为 None 则使用默认 HEAD_STRIDE8/16/32。
    """
    h8 = head_8 or HEAD_STRIDE8
    h16 = head_16 or HEAD_STRIDE16
    h32 = head_32 or HEAD_STRIDE32
    t = ByteTrackYoloXTransform(graph)
    split = SPLIT_CHANNELS if num_class == 1 else [4, 1, num_class]

    # Split each head (6 ch) -> loc(4), conf(1), cls(num_class)，按 ONNX 中实际名称连接
    t.AddSplitOp("split_8",  h8,  ["loc_8",  "conf_8",  "cls_8"],  axis=1, split=split)
    t.AddSplitOp("split_16", h16, ["loc_16", "conf_16", "cls_16"], axis=1, split=split)
    t.AddSplitOp("split_32", h32, ["loc_32", "conf_32", "cls_32"], axis=1, split=split)

    # YoloxDecoder_IXRT 插件输入顺序: (cls_prob, bbox, box_prob) -> [cls, loc, conf]
    t.AddYoloDecoderOp(
        "decoder_8", inputs=["cls_8", "loc_8", "conf_8"],
        outputs=["decoder_8_boxes", "decoder_8_scores"], num_class=num_class, stride=8
    )
    t.AddYoloDecoderOp(
        "decoder_16", inputs=["cls_16", "loc_16", "conf_16"],
        outputs=["decoder_16_boxes", "decoder_16_scores"], num_class=num_class, stride=16
    )
    t.AddYoloDecoderOp(
        "decoder_32", inputs=["cls_32", "loc_32", "conf_32"],
        outputs=["decoder_32_boxes", "decoder_32_scores"], num_class=num_class, stride=32
    )

    # Concat: order same as sampleYoloX (6400, 1600, 400 -> 8, 16, 32)
    t.AddConcatOp(
        "concat_boxes",
        inputs=["decoder_8_boxes", "decoder_16_boxes", "decoder_32_boxes"],
        outputs=["boxes"], axis=3
    )
    t.AddConcatOp(
        "concat_scores",
        inputs=["decoder_8_scores", "decoder_16_scores", "decoder_32_scores"],
        outputs=["scores"], axis=2
    )

    t.AddNMSOp(
        "nms", inputs=["boxes", "scores"],
        outputs=["num_detections", "detection_boxes", "detection_scores", "detection_classes"],
        share_location=1, iou_threshold=iou_threshold, score_threshold=score_threshold,
        max_output_boxes=1000, background_class=-1
    )

    t.DropUselessOutputs()
    t.Cleanup()
    return graph


def main():
    parser = argparse.ArgumentParser(description="Add Decoder+NMS to ByteTrack YOLOX ONNX")
    parser.add_argument("-i", "--src", required=True, help="Input ONNX (e.g. ../bytetrack_s.onnx)")
    parser.add_argument("-o", "--dest", required=True, help="Output ONNX (e.g. ../bytetrack_s_with_decoder_nms.onnx)")
    parser.add_argument("--num_class", type=int, default=1, help="Number of classes (default 1 for MOT)")
    parser.add_argument("--score_threshold", type=float, default=0.1, help="NMS score threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--max_ir_version", type=int, default=9, help="Max ONNX ir_version (downgrade if higher, for ixrt)")
    parser.add_argument("--output-name", default="output", help="ONNX 图中最终输出名，用于追溯 head 节点")
    parser.add_argument("--head-names", nargs=3, metavar=("HEAD_8", "HEAD_16", "HEAD_32"), default=None,
                        help="手动指定三个 head 张量名（stride 8,16,32），覆盖自动检测")
    parser.add_argument("--decoder-only", action="store_true",
                        help="仅添加 Decoder，输出 boxes 与 scores，不添加 NMS；用于与 demo_postprocess 对比验证 decoder 准确性")
    args = parser.parse_args()

    if not os.path.isfile(args.src):
        print("Error: input file not found:", args.src)
        sys.exit(1)

    # 从原始 ONNX 自动检测三个 head 节点名（保证连接正确），或使用 --head-names 手动指定
    if args.head_names:
        head_8, head_16, head_32 = args.head_names
        print("Using head names from --head-names:", head_8, head_16, head_32)
    else:
        heads = discover_head_names_from_onnx(args.src, output_name=args.output_name)
        if heads:
            head_8, head_16, head_32 = heads
        else:
            head_8, head_16, head_32 = HEAD_STRIDE8, HEAD_STRIDE16, HEAD_STRIDE32
            print("Using default head names:", head_8, head_16, head_32)

    print("Fixing ONNX for ixrt (ir_version, Conv kernel_shape, MaxPool/Resize drop unsupported attrs)...")
    onnx_path = _fix_onnx_for_ixrt(args.src, max_ir_version=args.max_ir_version)
    try:
        print("Loading", onnx_path)
        graph = create_source(onnx_path)()
    finally:
        if onnx_path != args.src and os.path.isfile(onnx_path):
            try:
                os.unlink(onnx_path)
            except Exception:
                pass
    if args.decoder_only:
        print("Adding Decoder only (num_class=%d), heads: %s, %s, %s" % (args.num_class, head_8, head_16, head_32))
        graph = add_decode_only(
            graph,
            num_class=args.num_class,
            head_8=head_8,
            head_16=head_16,
            head_32=head_32,
        )
    else:
        print("Adding Decoder + NMS (num_class=%d), heads: %s, %s, %s" % (args.num_class, head_8, head_16, head_32))
        graph = add_decode_nms(
            graph,
            num_class=args.num_class,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
            head_8=head_8,
            head_16=head_16,
            head_32=head_32,
        )
    print("Saving", args.dest)
    create_target(saved_path=args.dest).export(graph)
    print("Done. Final model:", args.dest)


if __name__ == "__main__":
    main()
