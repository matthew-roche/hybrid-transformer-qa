import tensorrt as trt
import pycuda.driver as cuda
# import pycuda.autoinit
import numpy as np
from transformers import AutoTokenizer, AutoConfig
import time, json, os

from add_path import add_project_path, model_output_path, data_path
add_project_path()

from util.helper import calculate_entropy_np, softmax_np, possibility_maybe, create_input_2d_array

load_attribute = "597_63_63"
opt_level = 5
engine_file_path = model_output_path() / "tensorrt" / f"classifier_{load_attribute}-{opt_level}.trt"

# init pycuda driver
cuda.init()
device = cuda.Device(0)
pycuda_context = device.make_context()

def load_tensorrt_classifier_engine():
    with open(engine_file_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    
    return engine, context

def load_tensorrt_classifier():
    with open(engine_file_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    filename = "cat2context_train.json"
    file_path = os.path.join(data_path(), filename)
    with open(file_path, "r", encoding="utf-8") as f:
        train_categeory_data_json = json.load(f)

    filename = f"english_dictionary_{load_attribute}.json"
    file_path = os.path.join(model_output_path(), filename)
    with open(file_path, "r", encoding="utf-8") as f:
        english_dictionary = json.load(f)

    filename = f"english_list_{load_attribute}.json"
    file_path = os.path.join(model_output_path(), filename)
    with open(file_path, "r", encoding="utf-8") as f:
        english_list = json.load(f)

    filename = "word_alias_mapper.json"
    file_path = os.path.join(data_path(), filename)
    with open(file_path, "r", encoding="utf-8") as f:
        word_alias_dictionary = json.load(f)
    
    context = engine.create_execution_context()
    
    return engine, context, english_dictionary, english_list, word_alias_dictionary, train_categeory_data_json


def prepare_context(engine, context, input_ids, attention_mask):
    seq_len = input_ids.shape[1]
    batch_size = len(input_ids) # 2d array item count

    # context = engine.create_execution_context() # reuse
    context.set_input_shape("input_ids", (batch_size, seq_len))
    context.set_input_shape("attention_mask", (batch_size, seq_len))

    # prepare mem alloc
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        #print(f"Binding: {binding}, Is Input: {engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT}")

        shape = context.get_tensor_shape(binding)  # Updated API
        dtype = trt.nptype(engine.get_tensor_dtype(binding))

        #print(shape)

        size = trt.volume(shape)  # Calculate total elements in the shape
        host_mem = cuda.pagelocked_empty(size, dtype)  # Host memory
        device_mem = cuda.mem_alloc(host_mem.nbytes)   # Device memory

        bindings.append(int(device_mem))

        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append({"host": host_mem, "device": device_mem})
        else:
            outputs.append({"host": host_mem, "device": device_mem})
    
    np.copyto(inputs[0]["host"], input_ids.flatten())
    np.copyto(inputs[1]["host"], attention_mask.flatten())

    cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
    cuda.memcpy_htod_async(inputs[1]["device"], inputs[1]["host"], stream)

    for i, binding in enumerate(engine):
        context.set_tensor_address(binding, bindings[i])

    return context, stream, outputs, host_mem, device_mem

def classifier_inference(question, english_dictionary, engine, context):
    question_array = create_input_2d_array(question, english_dictionary, add_pad_token=True)
    attention_mask = (question_array == 0)

    pycuda_context.push() # activate context
    context, stream, outputs, host_mem, device_mem = prepare_context(engine, context, question_array, attention_mask)

    inference_time = 0
    try:
        start_time = time.perf_counter()
        context.execute_async_v3(stream_handle=stream.handle) # inference

        for output in outputs:
            cuda.memcpy_dtoh_async(output["host"], output["device"], stream)
        
        stream.synchronize()
        end_time = time.perf_counter()
        inference_time = end_time - start_time
    except Exception as e:
        print(f"Execution Error: {e}")
    finally:
        del stream
        del host_mem
        del device_mem
        pycuda_context.pop() # release context
    
    return outputs, inference_time


def dynamic_context_trt(question, english_dictionary, engine, context, train_categeory_data_json):
    outputs, inference_time = classifier_inference(question, english_dictionary, engine, context)

    category_logits = outputs[0]['host']
    possibility_logits = outputs[1]["host"]

    category_predicted = np.argmax(category_logits)
    possibility_predicted = np.argmax(possibility_logits)
    entropy = calculate_entropy_np(possibility_logits)
    
    # if the classifier isn't confident, then infer 'maybe'
    if possibility_maybe(possibility_predicted, category_predicted, entropy):
        possibility_predicted = 2

    return possibility_predicted, train_categeory_data_json[category_predicted]['sentences'], inference_time

def context_label_trt(question, english_dictionary, engine, context, train_categeory_data_json):

    outputs, inference_time = classifier_inference(question, english_dictionary, engine, context)

    category_logits = outputs[0]['host']
    possibility_logits = outputs[1]["host"]

    category_predicted = np.argmax(category_logits)
    possibility_predicted = np.argmax(possibility_logits)

    possibility_softmax = softmax_np(possibility_logits)
    entropy = calculate_entropy_np(possibility_softmax)

    # if the classifier isn't confident, then infer 'maybe'
    if possibility_maybe(possibility_predicted, category_predicted, entropy):
        possibility_predicted = 2

    return possibility_predicted, train_categeory_data_json[category_predicted]['category_text'], inference_time
