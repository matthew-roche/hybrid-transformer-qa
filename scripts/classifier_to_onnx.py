import torch, os, json, time
import onnxruntime as ort
import numpy as np
import onnx
from torch.nn.utils.rnn import pad_sequence

from add_path import add_project_path, model_output_path, data_path, classifier_model_load_path
add_project_path()

from model.classifier_model import Classifier
from util.helper import create_input_tensor, create_input_array, create_input_2d_array

mode_export = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_attribute = "597_63_63"

filename = f"english_list_{load_attribute}.json"
file_path = os.path.join(model_output_path(), filename)
with open(file_path, "r", encoding="utf-8") as f:
    english_list = json.load(f)

filename = f"english_dictionary_{load_attribute}.json"
file_path = os.path.join(model_output_path(), filename)
with open(file_path, "r", encoding="utf-8") as f:
    english_disctionary = json.load(f)

filename = "context_mapper.json"
file_path = os.path.join(data_path(), filename)
with open(file_path, "r", encoding="utf-8") as f:
    context_label_mapper = json.load(f)


model = Classifier(max_vocab_size=len(english_list), embed_dim=512, category_count=len(context_label_mapper), possibility_count=2).to(device)
model.load_state_dict(torch.load(classifier_model_load_path(load_attribute)))
model.eval()


save_onnx_dir = model_output_path() / "onnx" / f"classifier_{load_attribute}.onnx"

def torch_predict(question_tensor, attention_mask):
    with torch.no_grad():
        category_logits, possibility_logits = model(question_tensor, attention_mask)
    model_category_predicted = torch.argmax(category_logits).item()
    model_possibility_predicted = torch.argmax(possibility_logits).item()

    return model_category_predicted, model_possibility_predicted

question_1 = "Do you allow to skip the first payment cycle ?"
question_1_tensor = create_input_tensor(question_1, english_disctionary, add_pad_token=True).unsqueeze(0).to(device)
attention_1_mask = (question_1_tensor == 0).to(device)

question_1_model_category_predicted, question_1_model_possibility_predicted = torch_predict(question_1_tensor, attention_1_mask)

if mode_export:
    if not os.path.exists(model_output_path() / "onnx"):
        os.makedirs(model_output_path() / "onnx")

    torch.onnx.export(
        model,
        (question_1_tensor, attention_1_mask),
        save_onnx_dir,
        input_names=["input_ids", "attention_mask"],
        output_names=["category", "possibility"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "category": {0: "batch_size", 1: "sequence_length"},
            "possibility": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=20,
        export_params=True,
        do_constant_folding=True,
        dynamo=False # for trt export later
    )

# verify onnx export

onnx_model = onnx.load(save_onnx_dir)
input_id = onnx_model.graph.input[0]
shape_info = input_id.type.tensor_type.shape

for i, dim in enumerate(shape_info.dim):
    print(f"Dim {i}: dim_value={dim.dim_value}, dim_param='{dim.dim_param}'")

def onnx_inference(question_test, word_disctionary):
    print(question_test)

    question_array = create_input_2d_array(question_test, word_disctionary, add_pad_token=True)
    attention_mask = (question_array == 0)

    start_time = time.time()
    onnx_input = {"input_ids": question_array, "attention_mask": attention_mask}
    category_logits, possibility_logits = ort_session.run(None, onnx_input)
    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.4f} seconds")

    category = np.argmax(category_logits[0])
    possibility = np.argmax(possibility_logits[0])

    return category, possibility

ort_session = ort.InferenceSession(save_onnx_dir)

category_1, possibility_1 = onnx_inference(question_1, english_disctionary)

# ensure onnx model performs the same as before export
assert category_1 == question_1_model_category_predicted
assert possibility_1 == question_1_model_possibility_predicted

# test against another input to ensure the model performs as before
question_2 = "My name was misspelled earlier, can I change it in my invoice ?"
question_2_tensor = create_input_tensor(question_2, english_disctionary, add_pad_token=True).unsqueeze(0).to(device)
attention_2_mask = (question_1_tensor == 0).to(device)

question_2_model_category_predicted, question_2_model_possibility_predicted = torch_predict(question_2_tensor, attention_2_mask)

category_2, possibility_2 = onnx_inference(question_2, english_disctionary)

assert category_2 == question_2_model_category_predicted
assert possibility_2 == question_2_model_possibility_predicted
