import onnx
from onnx import helper, TensorProto, shape_inference

def add_yolov8_postprocess(input_onnx_path, output_onnx_path):
    """
    修复：确保 output_scores 关联到 Split 节点的输出 output_scores，建立完整数据流向
    """
    # 1. 加载原始 ONNX 并推理形状
    model = onnx.load(input_onnx_path)
    inferred_model = shape_inference.infer_shapes(model)
    graph = inferred_model.graph
    print(f"加载原始模型：{input_onnx_path}")

    # 2. 定位原始输出 output0
    original_output = next((out for out in graph.output if out.name == "output0"), None)
    if not original_output:
        raise ValueError("未找到 output0 节点，请检查原始模型")

    # 3. 1. 添加 Split 节点（拆分 output0 为 temp_boxes 和 output_scores）
    # 3.1 创建 Split 拆分长度常量 [4,80]
    split_sizes = helper.make_tensor(
        name="Split_Sizes",
        data_type=TensorProto.INT64,
        dims=[2],
        vals=[4, 80]
    )
    graph.initializer.append(split_sizes)

    # 3.2 定义 Split 节点（输出 temp_boxes 和 output_scores，关键：这两个输出要被后续使用）
    split_node = helper.make_node(
        op_type="Split",
        inputs=["output0", "Split_Sizes"],  # 输入：原始 output0 + 拆分常量
        outputs=["temp_boxes", "output_scores"],  # 输出：两个临时张量（后续要关联到最终输出）
        name="Split_Node",
        axis=-2
    )
    graph.node.append(split_node)

    # 4. xywh2xyxy: x1 = x - w/2, y1 = y - h/2, x2 = x + w/2, y2 = y + h/2
    # 假设 temp_boxes 形状为 [1, 4, 4620]，顺序为 [x, y, w, h]

    # 4.1 添加常量
    graph.initializer.extend([
        helper.make_tensor("const_2", TensorProto.FLOAT, [1], [2.0]),
        helper.make_tensor("x_starts", TensorProto.INT64, [1], [0]),
        helper.make_tensor("x_ends", TensorProto.INT64, [1], [1]),
        helper.make_tensor("x_axes", TensorProto.INT64, [1], [1]),
        helper.make_tensor("y_starts", TensorProto.INT64, [1], [1]),
        helper.make_tensor("y_ends", TensorProto.INT64, [1], [2]),
        helper.make_tensor("y_axes", TensorProto.INT64, [1], [1]),
        helper.make_tensor("w_starts", TensorProto.INT64, [1], [2]),
        helper.make_tensor("w_ends", TensorProto.INT64, [1], [3]),
        helper.make_tensor("w_axes", TensorProto.INT64, [1], [1]),
        helper.make_tensor("h_starts", TensorProto.INT64, [1], [3]),
        helper.make_tensor("h_ends", TensorProto.INT64, [1], [4]),
        helper.make_tensor("h_axes", TensorProto.INT64, [1], [1]),
        helper.make_tensor("clamp_min", TensorProto.FLOAT, [1], [0.0]),
        helper.make_tensor("clamp_max", TensorProto.FLOAT, [1], [640.0]),
    ])

    # 4.2 Slice x, y, w, h
    x = helper.make_node("Slice", ["temp_boxes", "x_starts", "x_ends", "x_axes"], ["x"], name="Slice_x")
    y = helper.make_node("Slice", ["temp_boxes", "y_starts", "y_ends", "y_axes"], ["y"], name="Slice_y")
    w = helper.make_node("Slice", ["temp_boxes", "w_starts", "w_ends", "w_axes"], ["w"], name="Slice_w")
    h = helper.make_node("Slice", ["temp_boxes", "h_starts", "h_ends", "h_axes"], ["h"], name="Slice_h")
    graph.node.extend([x, y, w, h])

    # 4.3 计算 w/2, h/2
    w_half = helper.make_node("Div", ["w", "const_2"], ["w_half"], name="Div_w_half")
    h_half = helper.make_node("Div", ["h", "const_2"], ["h_half"], name="Div_h_half")
    graph.node.extend([w_half, h_half])

    # 4.4 x1 = x - w/2, x2 = x + w/2
    x1 = helper.make_node("Sub", ["x", "w_half"], ["x1"], name="Sub_x1")
    x2 = helper.make_node("Add", ["x", "w_half"], ["x2"], name="Add_x2")
    # 4.5 y1 = y - h/2, y2 = y + h/2
    y1 = helper.make_node("Sub", ["y", "h_half"], ["y1"], name="Sub_y1")
    y2 = helper.make_node("Add", ["y", "h_half"], ["y2"], name="Add_y2")
    graph.node.extend([x1, x2, y1, y2])

    # 4.6 Concat [x1, y1, x2, y2] (axis=1)
    boxes_xyxy = helper.make_node("Concat", ["x1", "y1", "x2", "y2"], ["boxes_xyxy"], name="Concat_xyxy", axis=1)
    graph.node.append(boxes_xyxy)

    # # 4.7 Clamp到[0, 640]
    # clamp_node = helper.make_node("Clip", ["boxes_xyxy", "clamp_min", "clamp_max"], ["boxes_xyxy_clamped"], name="Clip_xyxy")
    # graph.node.append(clamp_node)

    # 5. Reshape 节点（输入改为 boxes_xyxy_clamped）
    boxes_shape = helper.make_tensor(
        name="Boxes_Shape",
        data_type=TensorProto.INT64,
        dims=[4],
        vals=[-1, 4, 1, 4620]
    )
    graph.initializer.append(boxes_shape)

    reshape_node = helper.make_node(
        op_type="Reshape",
        inputs=["boxes_xyxy", "Boxes_Shape"],  # 输入改为 xyxy
        outputs=["output_boxes"],
        name="Reshape_Boxes"
    )
    graph.node.append(reshape_node)


    # 5. 3. 关键修复：更新模型输出（确保每个输出都有数据来源）
    del graph.output[:]  # 清空原始输出

    # 5.1 添加 output_boxes（来源：Reshape 节点的 output_boxes）
    graph.output.append(helper.make_tensor_value_info(
        name="output_boxes",
        elem_type=original_output.type.tensor_type.elem_type,
        shape=[-1, 4, 1, 4620]
    ))

    # 5.2 添加 output_scores（来源：Split 节点的 output_scores，这是之前缺失的关联）
    # 关键：输出节点名可以是 output_scores，但要明确它对应 Split 输出的 output_scores
    # 这里有两种等价方式，任选一种即可：

    # 方式1：直接将 Split 的 output_scores 作为 output_scores（推荐，更简洁）
    graph.output.append(helper.make_tensor_value_info(
        name="output_scores",  # 输出名与 Split 输出名一致，直接关联
        elem_type=original_output.type.tensor_type.elem_type,
        shape=[-1, 80, 4620]
    ))

    # 6. 验证模型
    try:
        onnx.checker.check_model(inferred_model)
        print(f"模型验证通过！新输出：{[out.name for out in graph.output]}")
    except onnx.checker.ValidationError as e:
        print(f"验证警告：{str(e)[:200]}...")

    # 7. 保存模型
    onnx.save(inferred_model, output_onnx_path)
    print(f"新模型保存至：{output_onnx_path}")


if __name__ == "__main__":
    add_yolov8_postprocess("./yolov8s_sim.onnx", "./yolov8s_sim_xyxy.onnx")
    