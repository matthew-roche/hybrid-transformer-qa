from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch, os
from scripts.context_loader import load_classifier, dynamic_context, context_label
from scripts.context_loader_trt import load_tensorrt_classifier_engine, dynamic_context_trt, context_label_trt
from scripts.add_path import fine_tuned_model_path
from util.helper import possibility_needed, sanitize_question, possiblity_label

save_m_dir = fine_tuned_model_path() / "deberta-v3-base-fine-tuned"
save_t_dir = fine_tuned_model_path() / "deberta-v3-base-fine-tuned-token"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def torch_device():
    return device

def load_qa_model():
    qa_model = AutoModelForQuestionAnswering.from_pretrained(save_m_dir)
    qa_tokenizer = AutoTokenizer.from_pretrained(save_t_dir)

    return qa_model, qa_tokenizer

# Load Classifier
def load_classifier_to_device():
    return load_classifier()

def classifier_inference(question, word_alias_dictionary, model, english_dictionary, english_list, train_categeory_data_json):
    sanitized_question = sanitize_question(question, word_alias_dictionary, english_list)
    possibility_needed_ = possibility_needed(question)

    possibility, context = context_label(sanitized_question, english_dictionary, model, train_categeory_data_json) # get the context based on classifier predicted category

    possibility_text = ""
    if possibility_needed_:
        possibility_text = possiblity_label(possibility)

    return possibility_text, context


def hybrid_inference(question, word_alias_dictionary, classifier_model, qa_model, qa_tokenizer, english_dictionary, english_list, train_categeory_data_json):
    sanitized_question = sanitize_question(question, word_alias_dictionary, english_list)
    possibility_needed_ = possibility_needed(question)

    possibility, context = dynamic_context(sanitized_question, english_dictionary, classifier_model, train_categeory_data_json) # get the context based on classifier predicted category

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

    model_device = next(qa_model.parameters()).device
    cuda_inputs = {key: value.to(model_device) for key, value in inputs.items()} # copy to gpu

    outputs = qa_model(**cuda_inputs) # inference on qa model

    start_logits = outputs.start_logits # start logits
    end_logits = outputs.end_logits # start logits
    start_index = torch.argmax(start_logits) # start index of context
    end_index = torch.argmax(end_logits) # end index of context

    qa_extracted_answer = qa_tokenizer.decode(inputs["input_ids"][0][start_index:end_index + 1]) # tokenizer can decode based on the context passed in inputs param

    possibility_text = ""
    if possibility_needed_:
        possibility_text = possiblity_label(possibility)

    return possibility_text, qa_extracted_answer


# Tensor rt

def load_tensorrt_classifier_to_device():
    return load_tensorrt_classifier_engine()

def classifier_inference_trt(question, word_alias_dictionary, engine, context, english_dictionary, english_list, train_categeory_data_json):
    
    sanitized_question = sanitize_question(question, word_alias_dictionary, english_list)
    possibility_needed_ = possibility_needed(question)

    possibility, context_label, inference_time = context_label_trt(sanitized_question, english_dictionary, engine, context, train_categeory_data_json) # get the context based on classifier predicted category

    possibility_text = ""
    if possibility_needed_:
        possibility_text = possiblity_label(possibility)

    return possibility_text, context_label