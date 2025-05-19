import torch, os, json
from add_path import add_project_path, data_path, model_output_path
add_project_path()

from model.classifier_model import Classifier
from util.helper import create_input_tensor, calculate_entropy, possibility_maybe

load_attribute = "591_68_75"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

def load_classifier():
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
    
    model_file_path = os.path.join(model_output_path(), f"classifier_{load_attribute}.pt")
    model = Classifier(max_vocab_size=len(english_list), embed_dim=512, category_count=6, possibility_count=2).to(device)
    model.load_state_dict(torch.load(model_file_path))
    model.eval()

    return model, english_dictionary, english_list, word_alias_dictionary, train_categeory_data_json

# format question in a way that only accepted vocab used in training is used, otherwise model doesn't understand
def sanitize_question(sentence, word_alias_dictionary, english_list):
    sentence = sentence.replace(".", " .")
    sentence = sentence.replace("?", " ?")
    words = sentence.split(" ")
    alias_words = list(word_alias_dictionary.keys())
    sanitized_words = []
    for word in words:
        if word not in english_list:
            if word in alias_words:
                sanitized_words.append(word_alias_dictionary[word])
        else:
            sanitized_words.append(word)

    return " ".join(sanitized_words)

def dynamic_context(question, english_dictionary, model, train_categeory_data_json):
    sentence_tensor = create_input_tensor(question, english_dictionary).unsqueeze(0).to(device)

    with torch.no_grad(): # no gradient calc
        category_logits, possibility_logits = model(sentence_tensor)
    
    category_predicted = torch.argmax(category_logits).item()
    possibility_predicted = torch.argmax(possibility_logits).item() # either 0-No or 1-Yes
    entropy = calculate_entropy(possibility_logits[0]).item()

    # if the classifier isn't confident, then infer 'maybe'
    if possibility_maybe(possibility_predicted, category_predicted, entropy):
        possibility_predicted = 2

    return possibility_predicted, train_categeory_data_json[category_predicted]['sentences']

def context_label(question, english_dictionary, model, train_categeory_data_json):
    sentence_tensor = create_input_tensor(question, english_dictionary).unsqueeze(0).to(device)

    with torch.no_grad(): # no gradient calc
        category_logits, possibility_logits = model(sentence_tensor)
    
    category_predicted = torch.argmax(category_logits).item()
    possibility_predicted = torch.argmax(possibility_logits).item() # either 0-No or 1-Yes
    entropy = calculate_entropy(possibility_logits[0]).item()

    # if the classifier isn't confident, then infer 'maybe'
    if possibility_maybe(possibility_predicted, category_predicted, entropy):
        possibility_predicted = 2

    return possibility_predicted, train_categeory_data_json[category_predicted]['category_text']

# Custom logic to analyze if Yes/No is needed
def possibility_needed(question):
    first_word = question.split(" ")[0]
    if first_word in ["what", "What", "Where", "where", "How", "how"]: # anyword that doesn't need Yes/No in answer
        return False
    
    return True