import os
import tensorrt as trt
from add_path import add_project_path, model_output_path
add_project_path()

# nvidia trt error Assertion !mValueMapUndo failed. Therefore cannot proceed to convert to trt
# issue url https://github.com/NVIDIA/TensorRT/issues/4288#issuecomment-2568304136

if not os.path.exists(model_output_path() / "tensorrt"):
    os.makedirs(model_output_path() / "tensorrt")

onnx_model_path = model_output_path() / "onnx" / f"deepset-deberta-v3-fine-tuned-squad2.onnx"

opt_level = 4
engine_file_path = model_output_path() / "tensorrt" / f"deepset-deBerta-v3-fine-tuned-squad2-{opt_level}.trt"

# TensorRT Logger
logger = trt.Logger(trt.Logger.WARNING)

# TensorRT Engine Builder
with trt.Builder(logger) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, logger) as parser:
    # Load ONNX model
    with open(onnx_model_path, "rb") as model_file:
        parser.parse(model_file.read())
    
    
    # Configure the builder
    config = builder.create_builder_config()
    # config.flags |= trt.BuilderFlag.CUDA_GRAPH
    # Set memory pool limits
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB workspace memory
    config.set_memory_pool_limit(trt.MemoryPoolType.TACTIC_SHARED_MEMORY, 1 << 28)  # 256 MB for tactic memory
    #config.flags = trt.BuilderFlag.FP16  # Enable FP16 precision
    config.builder_optimization_level = opt_level  # Set optimization level (0-5)

    # Ensure FP16 is disabled
    config.clear_flag(trt.BuilderFlag.FP16)

    # Ensure INT8 is disabled
    config.clear_flag(trt.BuilderFlag.INT8)

    profile = builder.create_optimization_profile()

    # Define input shapes (name must match the ONNX input names)
    profile.set_shape("input_ids", min=(1, 32), opt=(1, 128), max=(8, 512))  # Batch x Sequence
    profile.set_shape("attention_mask", min=(1, 32), opt=(1, 128), max=(8, 512))
    config.add_optimization_profile(profile)

    # Build the engine
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build the TensorRT engine")

    # Save the engine
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)

print(f"Engine saved to {engine_file_path}")