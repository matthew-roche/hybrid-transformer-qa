import onnxruntime as ort
import numpy as np
import time
from add_path import add_project_path, model_output_path, fine_tuned_model_path
add_project_path()

from transformers import AutoTokenizer, pipeline
from context_loader import load_context, load_inference_data
from util.helper import create_input_tensor, create_input_array, create_input_2d_array, sanitize_question, possibility_needed, possiblity_label

use_fine_tuned_debertav3 = True

load_attribute = "597_63_63"
model_save_onnx_dir = model_output_path() / "onnx" / f"classifier_{load_attribute}.onnx"
qa_save_onnx_dir = model_output_path() / "onnx" / f"deepset-deberta-v3-fine-tuned-squad2.onnx"
qa_tokenizer_dir = fine_tuned_model_path() / "deberta-v3-base-fine-tuned-token"

model_ort_session = ort.InferenceSession(model_save_onnx_dir)
qa_ort_session = ort.InferenceSession(qa_save_onnx_dir)

qa_tokenizer = AutoTokenizer.from_pretrained(qa_tokenizer_dir)

inference_test = load_inference_data()

english_dictionary, english_list, word_alias_dictionary, train_categeory_data_json = load_context()

# onnx classifier
def classifier_inference(question_text, word_dictionary):
    question_array = create_input_2d_array(question_text, word_dictionary, add_pad_token=True)
    attention_mask = (question_array == 0)

    onnx_input = {"input_ids": question_array, "attention_mask": attention_mask}
    category_logits, possibility_logits = model_ort_session.run(None, onnx_input)
    
    category = np.argmax(category_logits[0])
    possibility = np.argmax(possibility_logits[0])

    return category, possibility

def qa_inference(question_text, context_array):
    QA_input = {
        'question': question_text,
        'context': " ".join(context_array)
    }

    inputs = qa_tokenizer(
        QA_input['question'],
        QA_input['context'], 
        return_tensors="np", #numpy
        max_length=512,
        truncation=True
    )

    onnx_input = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}

    outputs = qa_ort_session.run(None, onnx_input)
    start_logits = outputs[0] # start logits in first arr position
    end_logits = outputs[1]

    start_index = np.argmax(start_logits, axis=1)[0]  # Batch index 0
    end_index = np.argmax(end_logits, axis=1)[0]

    return qa_tokenizer.decode(inputs["input_ids"][0][start_index:end_index + 1])

print("Starting ONNX inference session...")
process_start_time = time.time()
for question in inference_test:
    print(f"\n{question}")
    sanitized_question = sanitize_question(question, word_alias_dictionary, english_list)

    possibility_needed_ = possibility_needed(question)

    if __debug__:
        inference_start_time = time.time()
    
    category_predicted, possibility_predicted = classifier_inference(sanitized_question, english_dictionary)

    extracted_answer = " [ignored]"
    if use_fine_tuned_debertav3:
        context_array = train_categeory_data_json[category_predicted]['sentences'] # select array from train
        extracted_answer = qa_inference(question, context_array)
    
    if __debug__:
        inference_end_time = time.time()

    possibility_text = ""
    if possibility_needed_:
        possibility_text = possiblity_label(possibility_predicted)
    
    print(f"{possibility_text}{"," if possibility_needed_ else ""}{extracted_answer}")

    if __debug__:
        print(f"Question Inference time: {inference_end_time - inference_start_time:.4f} seconds")

process_end_time = time.time()
print(f"Total Inference time: {process_end_time - process_start_time:.4f} seconds")
