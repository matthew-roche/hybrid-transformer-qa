import onnxruntime as ort
import onnx, time
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch, os
import numpy as np

from add_path import add_project_path, model_output_path, fine_tuned_model_path
add_project_path()

export = True

save_m_dir = fine_tuned_model_path() / "deberta-v3-base-fine-tuned"
save_t_dir = fine_tuned_model_path() / "deberta-v3-base-fine-tuned-token"

save_onnx_dir = model_output_path() / "onnx" / f"deepset-deberta-v3-fine-tuned-squad2.onnx"

qa_model = AutoModelForQuestionAnswering.from_pretrained(save_m_dir)
qa_tokenizer = AutoTokenizer.from_pretrained(save_t_dir)
model_device = next(qa_model.parameters()).device

QA_input = {
    'question': "Can I get the ultra subscription ?",
    'context': "Users can subscribe to the starter plan to access basic features. The ultra subscription includes additional tools and insights. You can upgrade your current plan to ultra or premium anytime. The ultra plan includes analytics and advanced controls. You can trial the ultra plan before committing to it."
}

inputs = qa_tokenizer(
    QA_input['question'],
    QA_input['context'], 
    return_tensors="pt", 
    max_length=512,
    truncation=True
)

if export:
    cuda_inputs = {key: value.to(model_device) for key, value in inputs.items()} # copy to gpu

    torch.onnx.export(
        qa_model,  # PyTorch model
        (cuda_inputs["input_ids"], cuda_inputs["attention_mask"]),  # Model inputs
        save_onnx_dir,  # Output ONNX file
        input_names=["input_ids", "attention_mask"],  # Input layer names
        output_names=["output"],  # Output layer name
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"}
        },  # Dynamic axes for flexible input sizes
        opset_version=20,  # ONNX opset version compatible with TensorRT
    )

onnx_model = onnx.load(save_onnx_dir)
input_id = onnx_model.graph.input[0]
shape_info = input_id.type.tensor_type.shape

# check assigned shapes
for i, dim in enumerate(shape_info.dim):
    print(f"Dim {i}: dim_value={dim.dim_value}, dim_param='{dim.dim_param}'")

ort_session = ort.InferenceSession(save_onnx_dir)

start_time = time.time()
onnx_input = {"input_ids": inputs["input_ids"].cpu().numpy(), "attention_mask": inputs["attention_mask"].cpu().numpy()}
outputs = ort_session.run(None, onnx_input)
end_time = time.time()
print(f"Inference time: {end_time - start_time:.4f} seconds")

start_logits = outputs[0]  # Assuming first output corresponds to start_logits
end_logits = outputs[1]    # Assuming second output corresponds to end_logits

start_index = np.argmax(start_logits, axis=1)[0]  # Batch index 0
end_index = np.argmax(end_logits, axis=1)[0]

qa_extracted_answer = qa_tokenizer.decode(inputs["input_ids"][0][start_index:end_index + 1])
print(f"Question: {QA_input['question']}, Answer: {qa_extracted_answer}")