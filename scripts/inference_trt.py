from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch, time
from add_path import add_project_path, fine_tuned_model_path
add_project_path()

from context_loader import load_context, load_inference_data
from context_loader_trt import load_tensorrt_classifier, load_tensorrt_classifier_engine, dynamic_context_trt, context_label_trt 
from util.helper import sanitize_question, possibility_needed

save_m_dir = fine_tuned_model_path() / "deberta-v3-base-fine-tuned"
save_t_dir = fine_tuned_model_path() / "deberta-v3-base-fine-tuned-token"

qa_model = AutoModelForQuestionAnswering.from_pretrained(save_m_dir)
qa_tokenizer = AutoTokenizer.from_pretrained(save_t_dir)

model_device = next(qa_model.parameters()).device

classifier_engine, classifier_engine_context= load_tensorrt_classifier_engine()

english_dictionary, english_list, word_alias_dictionary, train_categeory_data_json = load_context()

inference_test = load_inference_data()

process_start_time = time.time()
for question in inference_test:
    sanitized_question = sanitize_question(question, word_alias_dictionary, english_list)
    possibility_needed_ = possibility_needed(question)

    if __debug__:
        inference_start_time = time.time()
    
    possibility_predicted, context, inference_time = dynamic_context_trt(sanitized_question, english_dictionary, classifier_engine, classifier_engine_context, train_categeory_data_json)

    QA_input = {
        'question': sanitized_question,
        'context': " ".join(context)
    }

    inputs = qa_tokenizer(
        QA_input['question'],
        QA_input['context'], 
        return_tensors="pt", 
        max_length=512,
        truncation=True 
    )

    cuda_inputs = {key: value.to(model_device) for key, value in inputs.items()} # copy to gpu

    outputs = qa_model(**cuda_inputs) # inference on qa model

    if __debug__:
        inference_end_time = time.time()

    start_logits = outputs.start_logits # start logits
    end_logits = outputs.end_logits # start logits
    start_index = torch.argmax(start_logits) # start index of context
    end_index = torch.argmax(end_logits) # end index of context

    qa_extracted_answer = qa_tokenizer.decode(inputs["input_ids"][0][start_index:end_index + 1]) # tokenizer can decode based on the context passed in inputs param

    print(f"\nQuestion: {question}\nAnswer: {"" if not possibility_needed_ else "Yes, " if possibility_predicted == 1 else "No, " if possibility_predicted == 0 else ""}{qa_extracted_answer}")

    if __debug__:
        print(f"Question Inference time: {inference_end_time - inference_start_time:.4f} seconds")

process_end_time = time.time()
print(f"Total Inference time: {process_end_time - process_start_time:.4f} seconds")