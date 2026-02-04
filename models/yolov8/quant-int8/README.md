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
```bash
bash create_engine_static.sh
```

