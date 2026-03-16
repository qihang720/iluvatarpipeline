#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a static-batch ONNX (e.g. batch=1 fixed) to a dynamic-batch ONNX.

Usage:
    python3 make_onnx_dynamic_batch.py -i bytetrack_s.onnx -o bytetrack_s_dynamic.onnx

The script:
  1. Loads the ONNX and marks the batch dimension of the model input/output as
     symbolic ("batch"), removing the fixed dim_value=1.
  2. Runs onnxsim (with overwrite_input_shapes batch=1) to simplify the graph
     using concrete H/W/C shapes, then restores the dynamic batch annotation on
     the final simplified model.
  3. Saves the result.  ixrtexec will then accept --min_shape/--opt_shape/--max_shape.

Requirements: onnx, onnxsim  (pip install onnx onnxsim)
"""

import argparse
import sys

try:
    import onnx
except ImportError:
    print("Error: please install onnx:  pip install onnx")
    sys.exit(1)

try:
    from onnxsim import simplify
except ImportError:
    print("Error: please install onnxsim:  pip install onnxsim")
    sys.exit(1)


def _set_dynamic(model: onnx.ModelProto, input_name: str, batch_sym: str = "batch"):
    """Mark the first (batch) dimension of input and all outputs as dynamic."""
    for inp in model.graph.input:
        if inp.name == input_name:
            dim = inp.type.tensor_type.shape.dim[0]
            dim.ClearField("dim_value")
            dim.dim_param = batch_sym
    for out in model.graph.output:
        if out.type.tensor_type.HasField("shape") and len(out.type.tensor_type.shape.dim) > 0:
            dim = out.type.tensor_type.shape.dim[0]
            dim.ClearField("dim_value")
            dim.dim_param = batch_sym


def main():
    parser = argparse.ArgumentParser(
        description="Patch a static-batch ONNX to have a dynamic batch dimension")
    parser.add_argument("-i", "--input",  required=True, help="Input ONNX (static batch)")
    parser.add_argument("-o", "--output", required=True, help="Output ONNX (dynamic batch)")
    parser.add_argument("--input-name", default="images", help="Name of the model input tensor")
    parser.add_argument("--input-h", type=int, default=608,  help="Input height (default 608)")
    parser.add_argument("--input-w", type=int, default=1088, help="Input width  (default 1088)")
    parser.add_argument("--no-simplify", action="store_true",
                        help="Skip onnxsim simplification step")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    model = onnx.load(args.input)

    # Check whether the batch dim is already dynamic.
    for inp in model.graph.input:
        if inp.name == args.input_name:
            d = inp.type.tensor_type.shape.dim[0]
            if d.dim_param:
                print(f"  Input '{args.input_name}' already has dynamic batch dim '{d.dim_param}'.")
                print(f"  Nothing to do — saving copy to {args.output}.")
                onnx.save(model, args.output)
                return

    if not args.no_simplify:
        print(f"  Running onnxsim (concrete shapes: {args.input_name}=[1,3,{args.input_h},{args.input_w}]) ...")
        model_sim, check = simplify(
            model,
            overwrite_input_shapes={args.input_name: [1, 3, args.input_h, args.input_w]},
        )
        if not check:
            print("  Warning: onnxsim validation failed; proceeding anyway.")
        model = model_sim

    print("  Setting batch dimension to dynamic ...")
    _set_dynamic(model, args.input_name)

    print(f"  Saving to {args.output} ...")
    onnx.save(model, args.output)
    print("Done.")

    # Quick sanity check
    for inp in model.graph.input:
        if inp.name == args.input_name:
            shape = [d.dim_param if d.dim_param else str(d.dim_value)
                     for d in inp.type.tensor_type.shape.dim]
            print(f"  Input shape: {shape}")


if __name__ == "__main__":
    main()
