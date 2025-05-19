import random, torch
from util.helper import add_words_to_dict, create_input_tensor

def train_data_processor(json_context_list, word_dictionary, word_list, pick_unique = True, device = 'cpu'):
    """
    Function to process category_context_train files, and create trainable sentence based object list

    Attributes:
        json_context_list (array): iterated to process categories
        pick_unique (bool): flag to check if sentence is unique, and skip clones, default is True
    """
    list = []
    unique_sentences = []
    context_label_mapper = {}
    for category in json_context_list:
        id = category['category']
        label = category['category_text']
        possibility = category['possibility']
        context_label_mapper[id] = label

        sentences = category['sentences']
        if len(sentences) > 0:
            for sentence in sentences:
                train_obj = {}
                if pick_unique:
                    if sentence in unique_sentences:
                        print(f"Warning: Category {id}, has non unique sentence: {sentence}")
                        continue # skip adding non unique sentence
                    unique_sentences.append(sentence)
                
                s = sentence.replace(".", " .")

                train_obj['category'] = id
                train_obj['possibility'] = possibility
                train_obj['sentence'] = sentence
                add_words_to_dict(word_dictionary, word_list, [s])

                sentence_tensor = create_input_tensor(s, word_dictionary)
                category_tensor = torch.tensor(id, dtype=torch.long)
                possibility_tensor = torch.tensor(possibility, dtype=torch.long)

                train_obj['sentence_tensor'] = sentence_tensor.to(device)
                train_obj['category_tensor'] = category_tensor.to(device)
                train_obj['possibility_tensor'] = possibility_tensor.to(device)

                list.append(train_obj)

    random.shuffle(list)

    return list, context_label_mapper, word_dictionary, word_list

def test_data_processor(json_question_list, word_dictionary, word_list, device):
    list = []
    for qa in json_question_list:
        question_sentence = qa['question']

        add_words_to_dict(word_dictionary, word_list, [question_sentence])
        question_tensor = create_input_tensor(question_sentence, word_dictionary)
        category_tensor = torch.tensor(qa['category'], dtype=torch.long)
        possibility_tensor = torch.tensor(qa['possibility'], dtype=torch.long)

        obj = {}
        obj['question'] = question_sentence
        obj['category'] = qa['category']
        obj['possibility'] = qa['possibility']

        obj['question_tensor'] = question_tensor.to(device)
        obj['category_tensor'] = category_tensor.to(device)
        obj['possibility_tensor'] = possibility_tensor.to(device)

        list.append(obj)
    
    random.shuffle(list)

    return list, word_dictionary, word_list