import os, json, torch, random, copy
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence

# to resolve path related imports below
from add_path import add_project_path, data_path, model_output_path
add_project_path()

from data.dataset_loader import train_data_processor, test_data_processor
from model.classifier_model import Classifier
from util.helper import plot_training_curves, add_words_to_dict

# check torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch device: {device}")

word_dictionary = {}
word_list = []
add_words_to_dict(word_dictionary, word_list, ['PAD']) # PAD_TOKEN in index 0

filename = "cat2context_train.json"
file_path = os.path.join(data_path(), filename)
with open(file_path, "r", encoding="utf-8") as f:
    train_categeory_data_json = json.load(f)

train_sentence_list, context_label_mapper, word_dictionary, word_list = train_data_processor(train_categeory_data_json, word_dictionary, word_list, device=device)
train_dataset_size = len(train_sentence_list)

print(f"Loaded {train_dataset_size} train sentences.")
for key, value in context_label_mapper.items():
    print(f"Category: {key} = {value}")

filename = "cat2context_test.json"
file_path = os.path.join(data_path(), filename)
with open(file_path, "r", encoding="utf-8") as f:
    test_question_data_json = json.load(f)

test_question_list, word_dictionary, word_list = test_data_processor(test_question_data_json, word_dictionary, word_list, device)
test_dataset_size = len(test_question_list)

print(f"Loaded {test_dataset_size} test questions.")

# context_mapper_file_path = os.path.join(data_path(), "context_mapper.json")
# with open(context_mapper_file_path, "w") as f:
#     json.dump(context_label_mapper, f)
# exit(0)

batch_size = 80
dataset ={
    'train' : np.array_split(train_sentence_list, batch_size), # batch training
    'test': test_question_list
}

print(f"Batch size: {batch_size}, each batch has {len(dataset['train'][0])} sentences.")

def train_model(model, optimizer, scheduler, dataset, criterion_category, criterion_pos, epochs = 10):
    training_curves = {} # for post training analysis on loss, and accuracy
    training_curves['train_category_loss'] = []
    training_curves['train_possibility_loss'] = []
    training_curves['test_category_loss'] = []
    training_curves['test_possibility_loss'] = []
    training_curves['test_category_acc'] = []
    training_curves['test_possibility_acc'] = []
    training_curves['test_acc'] = []
    final_accuracy = 0.00
    
    for epoch in range(epochs):
        model.train()
        train_cat_loss, train_pos_loss = 0, 0
        for idx, sentences_chunk in enumerate(dataset['train']):
            batch_array = []
            cat_array = []
            pos_array = []
            for sentence_obj in sentences_chunk:
                batch_array.append(sentence_obj['sentence_tensor'])
                cat_array.append(sentence_obj['category_tensor'])
                pos_array.append(sentence_obj['possibility_tensor'])
            
            sentence_batch = pad_sequence(batch_array, batch_first=True, padding_value=0) # pad token 0
            category_batch = torch.tensor(cat_array, dtype=torch.long).to(device)
            possibility_batch = torch.tensor(pos_array, dtype=torch.long).to(device) 

            optimizer.zero_grad()  # reset gradient params
            category_logits, possibility_logits = model(sentence_batch) # classification results as logits

            # batch loss calculation
            loss_cat = criterion_category(category_logits, category_batch)
            loss_pos = criterion_pos(possibility_logits, possibility_batch)
            (loss_cat+loss_pos).backward() # model track loss

            train_cat_loss+=loss_cat
            train_pos_loss+=loss_pos

            optimizer.step() # update model weights and biases

        training_curves['train_category_loss'].append(train_cat_loss.detach())
        training_curves['train_possibility_loss'].append(train_pos_loss.detach())

        # print(f"Epoch {epoch}, tain_loss_cat: {train_cat_loss.detach().item()}, train_loss_pos: {train_pos_loss.detach().item()}")

        # test model per epoch
        model.eval()
        optimizer.zero_grad() # reset gradient params
        test_cat_loss_total, test_pos_loss_total = 0, 0
        cat_correct, pos_correct = 0, 0
        for idx, test_item in enumerate(dataset['test']):
            question_tensor = test_item['question_tensor'].unsqueeze(0) #reshape from [..] to [[..]]
            category_tensor = test_item['category_tensor'].unsqueeze(0)
            possibility_tensor = test_item['possibility_tensor'].unsqueeze(0)

            category_expected = torch.tensor(test_item['category'], dtype=torch.long).to(device)
            possibility_expected = torch.tensor(test_item['possibility'], dtype=torch.long).to(device)

            with torch.no_grad():
                category_logits, possibility_logits = model(question_tensor)
                test_loss_cat = criterion_category(category_logits, category_tensor)
                test_loss_pos = criterion_pos(possibility_logits, possibility_tensor)

                test_cat_loss_total+=test_loss_cat
                test_pos_loss_total+=test_loss_pos

                category_predicted = torch.argmax(category_logits)
                possibility_predicted = torch.argmax(possibility_logits)

                if category_predicted == category_expected:
                    cat_correct+=1
                if possibility_expected == possibility_predicted:
                    pos_correct+=1
            
        
        training_curves['test_category_loss'].append(test_cat_loss_total.detach())
        training_curves['test_possibility_loss'].append(test_pos_loss_total.detach())

        training_curves['test_category_acc'].append(cat_correct)
        training_curves['test_possibility_acc'].append(pos_correct)

        scheduler.step()

        final_accuracy = round(((cat_correct + pos_correct)/(test_dataset_size*2))*100, 2)
        training_curves['test_acc'].append(final_accuracy)

        # print(f"Epoch {epoch}, train_loss_cat: {train_cat_loss.detach().item()}, train_loss_pos: {train_pos_loss.detach().item()}, test_cat_acc: {cat_correct}/{test_dataset_size}, test_pos_acc: {pos_correct}/{test_dataset_size}")
    
    return model, training_curves, final_accuracy

learn_rate = 0.00038
num_of_runs = 10
train_epochs = 20

# multi class classification with digit labels
criterion_category = torch.nn.CrossEntropyLoss()
criterion_pos = torch.nn.CrossEntropyLoss()

best_model_dict = {}
best_model_found = False
best_model_training_curves = {}
prev_accuracy = 0
best_accuracy = 50

for run in range(num_of_runs):
    print(f"Running Train iteration {run+1}")
    model = Classifier(max_vocab_size=len(word_list), embed_dim=512, category_count=len(context_label_mapper), possibility_count=2).to(device) # copy model to torch device
    optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate, fused=True) # instantiate weight, bias param updater
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.6) # reduce lr over epochs

    trained_model, training_curves, final_accuracy = train_model(model, optimizer, scheduler, dataset, criterion_category, criterion_pos, epochs=train_epochs)

    # get training values from last epoch (train_epochs-1)
    print(f"test_cat_acc: {training_curves['test_category_acc'][train_epochs-1]}/{test_dataset_size}, test_pos_acc: {training_curves['test_possibility_acc'][train_epochs-1]}/{test_dataset_size}, % accuracy: {final_accuracy}")

    if final_accuracy > prev_accuracy and final_accuracy > best_accuracy:
        best_model_dict = copy.deepcopy(trained_model.state_dict()) # clone, not reference
        best_model_training_curves = training_curves
        best_model_found = True
        best_accuracy = final_accuracy

        print(f"Found best model, caching model dictionary,,,")

    prev_accuracy = final_accuracy
    

if best_model_found:
    print("Loading best modal state dictionary...")
    model.load_state_dict(best_model_dict)

    #  sync tensor in gpu to cpu for saving to file
    best_model_training_curves['train_category_loss'] = torch.stack(best_model_training_curves['train_category_loss']).tolist()
    best_model_training_curves['train_possibility_loss'] = torch.stack(best_model_training_curves['train_possibility_loss']).tolist()
    best_model_training_curves['test_category_loss'] = torch.stack(best_model_training_curves['test_category_loss']).tolist()
    best_model_training_curves['test_possibility_loss'] = torch.stack(best_model_training_curves['test_possibility_loss']).tolist()

    last_category_correct = best_model_training_curves['test_category_acc'][train_epochs-1]
    last_possibility_correct = best_model_training_curves['test_possibility_acc'][train_epochs-1]
    last_category_accuracy = (last_category_correct/test_dataset_size)*100
    last_possibility_accuracy = (last_possibility_correct/test_dataset_size)*100

    print(f"test_cat_acc: {last_category_correct}/{test_dataset_size}, test_pos_acc: {last_possibility_correct}/{test_dataset_size}")

    # plot_training_curves(training_curves)
    # plt.show()

    cat_percent_r = str(last_category_accuracy).split(".")[0]
    pos_percent_r = str(last_possibility_accuracy).split(".")[0]

    model_save_dir = model_output_path() / f"classifier_{len(word_list)}_{cat_percent_r}_{pos_percent_r}.pt"
    torch.save(best_model_dict, model_save_dir)

    english_dictionary_file_path = model_output_path() / f"english_dictionary_{len(word_list)}_{cat_percent_r}_{pos_percent_r}.json"
    english_list_file_path = model_output_path() / f"english_list_{len(word_list)}_{cat_percent_r}_{pos_percent_r}.json"
    training_curves_file_path = model_output_path() / f"training_curves_{len(word_list)}_{cat_percent_r}_{pos_percent_r}.json"

    with open(english_dictionary_file_path, "w") as f:
        json.dump(word_dictionary, f)
    with open(english_list_file_path, "w") as f:
        json.dump(word_list, f)
    with open(training_curves_file_path, "w") as f:
        json.dump(best_model_training_curves, f)

    print(f"Saved as {model_save_dir}")
else:
    print("Best model not found !")










