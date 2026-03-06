# yolov8m模型
## 下载模型

```bash
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt
```
## Onnx
```bash
pip install ultralytics==8.2.51
python3 export.py --weight yolov8s.pt --batch 32
```

## Onnx2Ixrt
后续步骤集成到该脚本里
```bash
bash create_engine_static.sh
```

查看create_engine_static.sh，可在export时修改batchsize大小
```bash
python3 export.py --weight yolov8s.pt --batch 32
```

在量化使用quant.py时，请修改--bsz参数与一开始使用的batchsize保持一致
```bash
python3 quant.py --model_name YOLOV8 --model yolov8s_sim_xyxy.onnx --bsz 32 --dataset_dir /data/peiyuan.zhang/downloads/coco/images/val2017 --ann_file /data/peiyuan.zhang/downloads/coco/annotations/instances_val2017.json --observer hist_percentile --save_quant_model quantized_yolov8_bs32.onnx --disable_quant_names /model.22/Concat /model.22/Concat_1 /model.22/Concat_2 /model.22/Reshape /model.22/Reshape_1 /model.22/Reshape_2 /model.22/Concat_3 /model.22/Split /model.22/dfl/Reshape /model.22/dfl/Transpose /model.22/dfl/Softmax /model.22/dfl/Transpose_1 /model.22/dfl/conv/Conv /model.22/dfl/Reshape_1 /model.22/Slice /model.22/Slice_1 /model.22/Sub /model.22/Add_1 /model.22/Add_2 /model.22/Div_1 /model.22/Sub_1 /model.22/Concat_4 /model.22/Mul_2 /model.22/Sigmoid /model.22/Concat_5 Split_Node Slice_x Slice_w Slice_y Slice_h Div_w_half Div_h_half Sub_x1 Add_x2 Sub_y1 Add_y2 Concat_xyxy Reshape_Boxes --imgsz 640
```

## 图片测试
```bash
python3 inference_ixrt_dyn.py --model_engine yolov8s_sim_xyxy_withnms.engine  --input_image dog_640_352.jpg
```