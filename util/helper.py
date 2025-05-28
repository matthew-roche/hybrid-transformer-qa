import torch
import matplotlib.pyplot as plt
import numpy as np

# code from week 5 assignment
# function to use unique word vocab for embedding
def add_words_to_dict(word_dictionary, word_list, sentences):
    for sentence in sentences:
        for word in sentence.split(" "):
            if word in word_dictionary:
                continue
            else:
                word_list.append(word)
                word_dictionary[word] = len(word_list)-1

# function to get long tensor based on sentence words, in reference to word_dictionary
def create_input_tensor(sentence, word_dictionary, add_pad_token = False, pad_len = 64, pad_token = 0):
    words = sentence.split(" ")
    indices = [word_dictionary[word] for word in words]
    if add_pad_token and len(indices) < pad_len:
        padded = indices + [pad_token] * (pad_len - len(indices))
        return torch.tensor(padded, dtype=torch.long)

    return torch.tensor(indices, dtype=torch.long)#.unsqueeze(1)  # LongTensor

def create_input_2d_array(sentence, word_dictionary, add_pad_token = False, pad_len = 64, pad_token = 0):
    words = sentence.split(" ")
    indices = [word_dictionary[word] for word in words]
    if add_pad_token and len(indices) < pad_len:
        padded = indices + [pad_token] * (pad_len - len(indices))
        return np.array([padded], dtype=np.int64)
    
    return np.array([indices], dtype=np.int64)  # same as torch.long

def create_input_array(sentence, word_dictionary, add_pad_token = False, pad_len = 64, pad_token = 0):
    words = sentence.split(" ")
    indices = [word_dictionary[word] for word in words]
    if add_pad_token and len(indices) < pad_len:
        padded = indices + [pad_token] * (pad_len - len(indices))
        return np.array(padded, dtype=np.int64)
    
    return np.array(indices, dtype=np.int64)  # same as torch.long

# code from week 4 assignment, modified for category and possibility
def plot_training_curves(training_curves,
                         phases=['train_category', 'train_possibility', 'test_category', 'test_possibility'],
                         metrics=['loss']):
    epochs = list(range(len(training_curves['train_category_loss']))) # inferred
    for metric in metrics:
        plt.figure()
        plt.title(f'Training curves - {metric} of Best Run')
        for phase in phases:
            key = phase+'_'+metric
            if key in training_curves:
                # print(training_curves[phase+'_'+metric])
                scalar_values = training_curves[phase+'_'+metric]
                plt.plot(epochs, scalar_values)
        plt.xlabel('epoch')
        plt.legend(labels=phases)

def softmax_np(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)  # for numerical stability
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# define a simple entropy function
def calculate_entropy(probs):
    # Add small epsilon to avoid log(0)
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=0)

def calculate_entropy_np(probs):
    # Add small epsilon to avoid log(0)
    return -np.sum(probs * np.log(probs + 1e-10), axis=0)

def possibility_maybe(pos, cat, entropy, entropy_less = 0.6):
    # possibility doesn't match with relevant category
    if (pos == 0 and cat in [0, 2, 4]) or (pos == 1 and cat in [1, 3, 5]):
        return True
    
    # inconfident
    if entropy > entropy_less:
        return True

# refactored from context_loader.py
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

def possibility_needed(question):
    first_word = question.split(" ")[0]
    if first_word in ["what", "What", "Where", "where", "How", "how", "when", "When"]: # anyword that doesn't need Yes/No in answer
        return False
    
    return True

def possiblity_label(possibility):
    possibility_text = "" # for questions that doesn't need a yes/no
    if possibility == 1:
        possibility_text = "Yes"
    elif possibility == 0:
        possibility_text = "No"
    elif possibility == 2:
        possibility_text = "Maybe" # when model is inconfident

    return possibility_text