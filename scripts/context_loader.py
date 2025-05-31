import torch, os, json
from add_path import add_project_path, data_path, model_output_path
add_project_path()

from model.classifier_model import Classifier
from util.helper import create_input_tensor, calculate_entropy, possibility_maybe

load_attribute = "597_63_63"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

def load_inference_data():
    filename = "inference_test.json"
    file_path = os.path.join(data_path(), filename)
    with open(file_path, "r", encoding="utf-8") as f:
        inference_test = json.load(f)
        
    return inference_test

def load_context():
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
    
    return english_dictionary, english_list, word_alias_dictionary, train_categeory_data_json

def load_classifier():
    english_dictionary, english_list, word_alias_dictionary, train_categeory_data_json = load_context()
    
    model_file_path = os.path.join(model_output_path(), f"classifier_{load_attribute}.pt")
    model = Classifier(max_vocab_size=len(english_list), embed_dim=512, category_count=6, possibility_count=2).to(device)
    model.load_state_dict(torch.load(model_file_path))
    model.eval()

    return model, english_dictionary, english_list, word_alias_dictionary, train_categeory_data_json

def dynamic_context(question, english_dictionary, model, train_categeory_data_json):
    sentence_tensor = create_input_tensor(question, english_dictionary).unsqueeze(0).to(device)
    attention_mask = (sentence_tensor == 0).to(device)

    with torch.no_grad(): # no gradient calc
        category_logits, possibility_logits = model(sentence_tensor, attention_mask)
    
    category_predicted = torch.argmax(category_logits).item()
    possibility_predicted = torch.argmax(possibility_logits).item() # either 0-No or 1-Yes
    entropy = calculate_entropy(possibility_logits[0]).item()

    # if the classifier isn't confident, then infer 'maybe'
    if possibility_maybe(possibility_predicted, category_predicted, entropy):
        possibility_predicted = 2

    return possibility_predicted, train_categeory_data_json[category_predicted]['sentences']

def context_label(question, english_dictionary, model, train_categeory_data_json):
    sentence_tensor = create_input_tensor(question, english_dictionary).unsqueeze(0).to(device)
    attention_mask = (sentence_tensor == 0).to(device)

    with torch.no_grad(): # no gradient calc
        category_logits, possibility_logits = model(sentence_tensor, attention_mask)
    
    category_predicted = torch.argmax(category_logits).item()
    possibility_predicted = torch.argmax(possibility_logits).item() # either 0-No or 1-Yes
    entropy = calculate_entropy(possibility_logits[0]).item()

    # if the classifier isn't confident, then infer 'maybe'
    if possibility_maybe(possibility_predicted, category_predicted, entropy):
        possibility_predicted = 2

    return possibility_predicted, train_categeory_data_json[category_predicted]['category_text']