import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import numpy as np
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--src", help="The exported to ONNX YOLOX-m", type=str,
    )
    parser.add_argument(
        "-o", "--dest", help="The output ONNX model file to write", type=str,
    )
    parser.add_argument(
        "--fusion_names", nargs='+', type=str, help="Two output node names: [Mul.90, Concat.29]"
    )
    parser.add_argument(
        "-m", "--max_output_boxes", help="Max output boxes for the NMS operation", type=int, default=1000,
    )
    parser.add_argument(
        "-st", "--score_threshold", help="The scalar threshold for score", type=float, default=0.7,
    )
    parser.add_argument(
        "-it", "--iou_threshold", help="The scalar threshold for IOU", type=float, default=0.45,
    )

    args = parser.parse_args()
    if not all([args.src, args.dest, args.fusion_names]):
        parser.print_help()
        print(
            "\nThese arguments are required: --src --dest --fusion_names"
        )
        sys.exit(1)
    print(args)
    return args

def modify_model(args):
    model = onnx.load(args.src)
    graph = model.graph

    # 1. 定义NMS_IXRT节点
    nms_output_names = [
            "num_detections",
            "detection_boxes",
            "detection_scores",
            "detection_classes"
        ]
    nms_node = helper.make_node(
        "NMS_IXRT",
        inputs=[args.fusion_names[0], args.fusion_names[1]],
        outputs=nms_output_names,
        name="nms",
        background_class=-1,
        iou_threshold=args.iou_threshold,
        max_output_boxes=args.max_output_boxes,
        score_threshold=args.score_threshold,
        share_location=1
    )
    graph.node.extend([nms_node])

    # 2. 处理 detection_classes 和 detection_scores 的 Unsqueeze 操作
    unsq_classes = helper.make_node(
        "Unsqueeze",
        inputs=[nms_output_names[3]],
        outputs=["detection_classes_unsq"],
        name="Unsqueeze_classes",
        axes=[-1]
    )
    unsq_scores = helper.make_node(
        "Unsqueeze",
        inputs=[nms_output_names[2]],
        outputs=["detection_scores_unsq"],
        name="Unsqueeze_scores",
        axes=[-1]
    )
    graph.node.extend([unsq_classes, unsq_scores])

    # 3. 拼接得到 nms_output0
    concat_node = helper.make_node(
        "Concat",
        inputs=[nms_output_names[1], "detection_classes_unsq", "detection_scores_unsq"],
        outputs=["nms_output0"],
        name="Concat_nms_output0",
        axis=2
    )
    graph.node.append(concat_node)

    # 4. 修复 num_detections 的处理：float32(batch,1) → int32(batch)
    # 4.1 Cast：float32 转 int32
    cast_node = helper.make_node(
        "Cast",
        inputs=[nms_output_names[0]],
        outputs=["num_detections_cast"],
        name="Cast_num_detections",
        to=TensorProto.INT32
    )

    # 4.2 关键修复：创建常量节点存储 axes，通过输入传递给 Squeeze（而非属性）
    # 定义 axes 常量（值为1，对应要删除的维度）
    axes_tensor = helper.make_tensor(
        name="squeeze_axes_const",  # 常量节点名称
        data_type=TensorProto.INT64,  # axes 必须是 INT64 类型
        dims=[1],  # 张量形状：1个元素
        vals=[1]   # 具体值：删除第1维（索引从0开始，(batch,1) → 删除后为 (batch)）
    )
    # 将常量节点添加到图的初始izer中
    graph.initializer.append(axes_tensor)

    # 4.3 Squeeze：通过输入传递 axes，而非属性
    squeeze_node = helper.make_node(
        "Squeeze",
        inputs=["num_detections_cast", "squeeze_axes_const"],  # 输入：待处理张量 + 常量axes
        outputs=["int_output1"],
        name="Squeeze_num_detections"
        # 移除原来的 axes 属性，避免TensorRT解析冲突
    )

    # 将 Cast 和 修复后的 Squeeze 节点添加到图中
    graph.node.extend([cast_node, squeeze_node])

    # 5. 设置模型输出
    graph.output.clear()
    # 输出1：处理后的 num_detections（int32，形状 (batch,)）
    
    # 输出2：拼接后的检测结果
    graph.output.append(
        helper.make_tensor_value_info("nms_output0", TensorProto.FLOAT, [None, args.max_output_boxes, 6])
    )

    graph.output.append(
        helper.make_tensor_value_info("int_output1", TensorProto.INT32, [None])
    )
    
    # 推理形状并保存模型
    # model = shape_inference.infer_shapes(model)
    onnx.save(model, args.dest)
    print(f"Modified model saved to: {args.dest}")

if __name__ == "__main__":
    args = parse_args()
    modify_model(args)