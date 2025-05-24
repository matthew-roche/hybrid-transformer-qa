import torch, os, json, random, time, pathlib, sys
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from add_path import add_project_path, model_output_path, data_path, huggingface_model_path, classifier_model_load_path
add_project_path()

from model.classifier_model import Classifier
from util.helper import plot_training_curves, create_input_tensor, calculate_entropy, possibility_maybe

load_attribute = "597_63_63"
use_qa_model = True # use answer extraction model alongside the trained classifier

qa_model_dir = huggingface_model_path() / "deberta-v3-base-squad2"
qa_tokenizer_dir = huggingface_model_path() / "deberta-v3-base-squad2-token"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

filename = "context_mapper.json"
file_path = os.path.join(data_path(), filename)
with open(file_path, "r", encoding="utf-8") as f:
    context_label_mapper = json.load(f)

for key, value in context_label_mapper.items():
    print(f"Category: {key} = {value}")

#  for deberta answer extraction
filename = "cat2context_train.json"
file_path = os.path.join(data_path(), filename)
with open(file_path, "r", encoding="utf-8") as f:
    train_categeory_data_json = json.load(f)

filename = "cat2context_test.json"
file_path = os.path.join(data_path(), filename)
with open(file_path, "r", encoding="utf-8") as f:
    test_dataset = json.load(f)

# random.shuffle(test_dataset) # can be toggled to change perceptions

filename = f"english_dictionary_{load_attribute}.json"
file_path = os.path.join(model_output_path(), filename)
with open(file_path, "r", encoding="utf-8") as f:
    english_disctionary = json.load(f)

filename = f"english_list_{load_attribute}.json"
file_path = os.path.join(model_output_path(), filename)
with open(file_path, "r", encoding="utf-8") as f:
    english_list = json.load(f)

filename = f"training_curves_{load_attribute}.json"
file_path = os.path.join(model_output_path(), filename)
with open(file_path, "r", encoding="utf-8") as f:
    training_curves = json.load(f)

model = Classifier(max_vocab_size=len(english_list), embed_dim=512, category_count=len(context_label_mapper), possibility_count=2).to(device)
model.load_state_dict(torch.load(classifier_model_load_path(load_attribute)))
model.eval() # set model to evaluation mode for testing

if use_qa_model:
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_dir)
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_tokenizer_dir)

    model_device = next(qa_model.parameters()).device

temperature = 0.9

# To analyze the model more deeper against test dataset
experiment_curves = {}
experiment_curves['category_max_softmax'] = []
experiment_curves['category_min_softmax'] = []
experiment_curves['category_expected_sotmax'] = []

experiment_curves['possibility_max_softmax'] = []
experiment_curves['possibility_min_softmax'] = []
experiment_curves['possibility_expected_softmax'] = []

print(f"Results for all test question: \n")
for question_item in test_dataset:
    question = question_item['question']
    question_tensor = create_input_tensor(question, english_disctionary, add_pad_token=True).unsqueeze(0).to(device)
    attention_mask = (question_tensor == 0).to(device)

    with torch.no_grad():
        category_logits, possibility_logits = model(question_tensor, attention_mask)

    
    cat_probs = F.softmax(category_logits[0], dim=0).cpu()
    pos_probs = F.softmax(possibility_logits[0], dim=0).cpu()

    entropy = calculate_entropy(possibility_logits[0]).item()

    #  most activated
    category_predicted = torch.argmax(category_logits).item()
    possibility_predicted = torch.argmax(possibility_logits).item()

    category_expected = question_item['category']
    possibility_expected = question_item['possibility']

    # least activated
    category_argmin = torch.argmin(category_logits).item()
    pos_argmin = torch.argmin(possibility_logits).item()

    # append the values to experiment curves array for plotting purpose
    experiment_curves['category_max_softmax'].append(cat_probs[category_predicted])
    experiment_curves['category_min_softmax'].append(cat_probs[category_argmin])
    experiment_curves['category_expected_sotmax'].append(cat_probs[category_expected])

    experiment_curves['possibility_max_softmax'].append(pos_probs[possibility_predicted])
    experiment_curves['possibility_min_softmax'].append(pos_probs[pos_argmin])
    experiment_curves['possibility_expected_softmax'].append(pos_probs[possibility_expected])

    if possibility_maybe(possibility_predicted, category_predicted, entropy):
        print(f"Inferred possibility: maybe") # for analysis

    if not use_qa_model:
        print(f"{question}\nCategory: {context_label_mapper[str(category_predicted)]} {"[incorrect]" if category_expected != category_predicted else ""}\n{"Yes" if possibility_predicted == 1 else "No"} {"[incorrect]" if possibility_expected != possibility_predicted else ""}\n")

    if use_qa_model:
        context = train_categeory_data_json[category_predicted]['sentences'] # because category id is also the index of array item

        QA_input = {
            'question': question,
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

        start_time = time.time()
        outputs = qa_model(**cuda_inputs)
        end_time = time.time()

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # start_probs = F.softmax(start_logits / temperature, dim=-1)
        # end_probs = F.softmax(end_logits / temperature, dim=-1)

        # start_index = torch.multinomial(start_probs, 1).item()
        # end_index = torch.multinomial(end_probs, 1).item()

        start_index = torch.argmax(start_logits)
        end_index = torch.argmax(end_logits)

        print(f"\nQA Inference time: {end_time - start_time:.4f} seconds,")

        answer = qa_tokenizer.decode(inputs["input_ids"][0][start_index:end_index + 1])
        if "[CLS]" in answer:
            answer = answer.split("[CLS]")[0]

        print(question)
        print(f"Answer: {"Yes, " if possibility_predicted == 1 else "No, "}{answer}")

# show loss graph from training curves
plot_training_curves(training_curves)

# Understand test dataset behaviour against the classification model
plt.figure()
plt.title(f'Correct predictions analysis, out of {len(test_dataset)}')
len_list = range(len(training_curves['test_possibility_acc']))
plt.plot(len_list, training_curves['test_category_acc'])
plt.plot(len_list, training_curves['test_possibility_acc'])
plt.xlabel('test sentences')
plt.legend(labels=['category_correct','possibility_correct'])
plt.grid(True, axis = 'y')

# Analyze any possible guard implementations
plt.figure()
plt.title(f'Category Softmax Analysis')
len_list = range(len(experiment_curves['category_max_softmax']))
plt.plot(len_list, experiment_curves['category_max_softmax'])
# plt.plot(len_list, experiment_curves['category_min_softmax'])
plt.plot(len_list, experiment_curves['category_expected_sotmax'])
plt.xlabel('test sentences')
plt.legend(labels=['cat_predicted','cat_expected'])
plt.grid(True, axis = 'y')

plt.figure()
plt.title(f'Possibility Softmax Analysis')
len_list = range(len(experiment_curves['possibility_expected_softmax']))
plt.plot(len_list, experiment_curves['possibility_max_softmax'])
# plt.plot(len_list, experiment_curves['possibility_min_softmax'])
plt.plot(len_list, experiment_curves['possibility_expected_softmax'])
plt.xlabel('test sentences')
plt.legend(labels=['pos_predicted','pos_expected'])
plt.grid(True, axis = 'y')

plt.show()