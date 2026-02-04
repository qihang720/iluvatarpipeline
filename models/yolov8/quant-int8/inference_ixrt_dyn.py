#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os
import time
import sys

import torch
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

from tqdm import tqdm
from tqdm.contrib import tzip

import tensorrt
from tensorrt import Dims
import cv2

import ctypes
from os.path import join, dirname, exists

def load_ixrt_plugin(logger=tensorrt.Logger(tensorrt.Logger.INFO), namespace="", dynamic_path=""):
    if not dynamic_path:
        # dynamic_path = join(dirname(tensorrt.__file__), "lib", "libixrt_plugin.so")
        # dynamic_path = join(dirname(tensorrt.__file__), "lib", "liboss_ixrt_plugin.so")
        dynamic_path = "/usr/local/corex/lib/liboss_ixrt_plugin.so"
    if not exists(dynamic_path):
        raise FileNotFoundError(
            f"The ixrt_plugin lib {dynamic_path} is not existed, please provided effective plugin path!")
    print(dynamic_path)
    ctypes.CDLL(dynamic_path)
    tensorrt.init_libnvinfer_plugins(logger, namespace)
    print(f"Loaded plugin from {dynamic_path}")

def create_engine_context(engine_path, logger):
    with open(engine_path, "rb") as f:
        runtime = tensorrt.Runtime(logger)
        assert runtime
        engine = runtime.deserialize_cuda_engine(f.read())
        assert engine
        context = engine.create_execution_context()
        assert context

    return engine, context

def setup_io_bindings(engine, context):
    # Setup I/O bindings
    inputs = []
    outputs = []
    allocations = []

    for i in range(engine.num_bindings):
        is_input = False
        if engine.binding_is_input(i):
            is_input = True
        name = engine.get_binding_name(i)
        dtype = engine.get_binding_dtype(i)
        shape = context.get_binding_shape(i)
        if is_input:
            batch_size = shape[0]
        size = np.dtype(tensorrt.nptype(dtype)).itemsize
        for s in shape:
            size *= s
        print("input dtype ", dtype)
        allocation = cuda.mem_alloc(size)
        binding = {
            "index": i,
            "name": name,
            "dtype": np.dtype(tensorrt.nptype(dtype)),
            "shape": list(shape),
            "allocation": allocation,
        }
        allocations.append(allocation)
        if engine.binding_is_input(i):
            inputs.append(binding)
        else:
            outputs.append(binding)
        print(shape)
    return inputs, outputs, allocations

def preprocess(image_path):
    image_ori = cv2.imread(image_path)
    image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, (640, 352))  # Assuming the input size is 352
    normalized_image = resized_image / 255.0
    transposed_image = np.transpose(normalized_image, (2, 0, 1))
    input_tensor = np.expand_dims(transposed_image, axis=0).astype(np.float32)

    # input_tensor = np.tile(transposed_image, (32,) + (1,) * (len(transposed_image.shape))).astype(np.float32)
    

    return input_tensor, image_ori

def postprocess(nms_output0, nms_output1, conf_threshold=0.4):
    numbers = nms_output1[0]
    predictions = nms_output0[0]  # Assuming outputs[0] is the output tensor
    boxes = []
    confidences = []
    class_ids = []

    for i in range(numbers):
        pred = predictions[i]  # Confidence score is in the 7th position
        confidence = pred[-1]
        if confidence > conf_threshold:
            class_id = int(pred[4])  # Class ID is in the 6th position
            x1 = int(pred[0])
            y1 = int(pred[1])
            x2 = int(pred[2])
            y2 = int(pred[3])
            boxes.append([x1, y1, x2, y2])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            print(f"class_id:{class_id}, confidence:{confidence}, x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}")

    return boxes, confidences, class_ids

def draw_boxes(image, boxes, confidences, class_ids):
    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"ID: {class_id}, Conf: {confidence:.2f}"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main(config):
    load_ixrt_plugin()

    input_tensor, image = preprocess(config.input_image)

    host_mem = tensorrt.IHostMemory
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)

    # Load Engine
    engine, context = create_engine_context(config.model_engine, logger)
    
    # set dyn batch
    input_idx = engine.get_binding_index("images")
    print(f"input shape : {input_tensor.shape} input type : {input_tensor.dtype}")
    # context.set_binding_shape(input_idx, Dims(input_tensor.shape))
    context.set_binding_shape(input_idx, Dims((input_tensor.shape)))
    inputs, outputs, allocations = setup_io_bindings(engine, context)

    # Prepare the output data
    output = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])
    print(f"output shape : {output.shape} output type : {output.dtype}")
    output = np.zeros(outputs[1]["shape"], outputs[1]["dtype"])
    print(f"output shape : {output.shape} output type : {output.dtype}")
    nms_output0 = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])
    nms_output1 = np.zeros(outputs[1]["shape"], outputs[1]["dtype"]) 

    # Set input
    cuda.memcpy_htod(inputs[0]["allocation"], np.ascontiguousarray(input_tensor))

    # Forward
    torch.cuda.synchronize()
    context.execute_v2(allocations)
    torch.cuda.synchronize()
    
    # output
    cuda.memcpy_dtoh(nms_output0, outputs[0]["allocation"])
    cuda.memcpy_dtoh(nms_output1, outputs[1]["allocation"])
    
    print("nms_output0 shape:", nms_output0.shape)
    print("nms_output1 shape:", nms_output1.shape)
    print("nms_output1:", nms_output1)
    
    boxes, confidences, class_ids = postprocess(nms_output0, nms_output1, conf_threshold=0.25)

    draw_boxes(image, boxes, confidences, class_ids)

    # Save or display the output image
    cv2.imwrite('output.jpg', image)

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_engine",
        type=str,
        default="",
        help="model engine path",
    )
    parser.add_argument(
        "--input_image",
        type=str,
        default="",
        help="test images",
    )
    config = parser.parse_args()
    print("config:", config)
    return config

if __name__ == "__main__":
    config = parse_config()
    main(config)