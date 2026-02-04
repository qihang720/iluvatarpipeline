python3 export.py --weight yolov8s.pt --batch 32
onnxsim ./yolov8s.onnx ./yolov8s_sim.onnx 
# --overwrite-input-shape "images:1,3,352,640"
python3 ./add_xywh2xyxy_onnx_352.py

python3 quant.py --model_name YOLOV8 --model yolov8s_sim_xyxy.onnx --bsz 32 --dataset_dir /data/peiyuan.zhang/downloads/coco/images/val2017 --ann_file /data/peiyuan.zhang/downloads/coco/annotations/instances_val2017.json --observer hist_percentile --save_quant_model quantized_yolov8_bs32.onnx --disable_quant_names /model.22/Concat /model.22/Concat_1 /model.22/Concat_2 /model.22/Reshape /model.22/Reshape_1 /model.22/Reshape_2 /model.22/Concat_3 /model.22/Split /model.22/dfl/Reshape /model.22/dfl/Transpose /model.22/dfl/Softmax /model.22/dfl/Transpose_1 /model.22/dfl/conv/Conv /model.22/dfl/Reshape_1 /model.22/Slice /model.22/Slice_1 /model.22/Sub /model.22/Add_1 /model.22/Add_2 /model.22/Div_1 /model.22/Sub_1 /model.22/Concat_4 /model.22/Mul_2 /model.22/Sigmoid /model.22/Concat_5 Split_Node Slice_x Slice_w Slice_y Slice_h Div_w_half Div_h_half Sub_x1 Add_x2 Sub_y1 Add_y2 Concat_xyxy Reshape_Boxes --imgsz 640

python3 ./customize_op_for_model.py \
        --src quantized_yolov8_bs32.onnx \
        --dest ./yolov8s_sim_xyxy_withnms.onnx \
        --fusion_names "output_boxes" "output_scores" \
        --max_output_boxes 1000 \
        --score_threshold 0.5 \
        --iou_threshold 0.5

ixrtexec --onnx ./yolov8s_sim_xyxy_withnms.onnx \
        --precision int8 \
        --plugins /usr/local/corex/lib/liboss_ixrt_plugin.so \
        --save_engine ./yolov8s_sim_xyxy_withnms.engine 

# ixrtexec --onnx ./yolov8s_sim_xyxy_withnms.onnx \
#         --precision int8 \
#         --min_shape images:1x3x640x640 \
#         --opt_shape images:16x3x640x640 \
#         --max_shape images:64x3x640x640 \
#         --plugins /usr/local/corex/lib/liboss_ixrt_plugin.so \
#         --save_engine ./yolov8s_sim_xyxy_withnms.engine 