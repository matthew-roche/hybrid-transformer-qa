import torch
import matplotlib.pyplot as plt

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
def create_input_tensor(sentence, word_dictionary):
    words = sentence.split(" ")
    indices = [word_dictionary[word] for word in words]
    return torch.tensor(indices, dtype=torch.long)#.unsqueeze(1)  # LongTensor

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

# define a simple entropy function
def calculate_entropy(probs):
    # Add small epsilon to avoid log(0)
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=0)

def possibility_maybe(pos, cat, entropy, entropy_less = 0.6):
    # possibility doesn't match with relevant category
    if (pos == 0 and cat in [0, 2, 4]) or (pos == 1 and cat in [1, 3, 5]):
        return True
    
    # inconfident
    if entropy > entropy_less:
        return True