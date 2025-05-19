from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments, DefaultDataCollator
import time, os
import torch
import torch.nn.functional as F
from datasets import load_dataset
import matplotlib.pyplot as plt
from add_path import data_path, huggingface_model_path, fine_tuned_model_path

# https://huggingface.co/learn/llm-course/en/chapter3/3
# https://huggingface.co/docs/transformers/en/main_classes/trainer

save = True # download model locally

# 2nd model from hugging face, trained on squard 2 for answer extraction
model_name = "deepset/deberta-v3-base-squad2"
save_m_dir = huggingface_model_path() / "deberta-v3-base-squad2"
save_t_dir = huggingface_model_path() / "deberta-v3-base-squad2-token"

if save:
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.save_pretrained(save_m_dir)
    tokenizer.save_pretrained(save_t_dir)
    print(f"Model saved to {save_m_dir}")

print(f"Loading model from {save_m_dir}")
start_time = time.time()
model = AutoModelForQuestionAnswering.from_pretrained(save_m_dir)
tokenizer = AutoTokenizer.from_pretrained(save_t_dir)
end_time = time.time()
print(f"Model load time: {end_time - start_time:.4f} seconds")

deberta_train_json = file_path = os.path.join(data_path(), "deberta_train.json")

data = load_dataset("json", data_files={"train": deberta_train_json})

# https://huggingface.co/docs/transformers/en/tasks/question_answering
# fill the dataset in data['train']
def preprocess(sample):
    return tokenizer(
        sample["question"],
        sample["context"],
        truncation="only_second",
        max_length=512,
        stride=32,
        return_offsets_mapping=True,
        padding="max_length"
    )

# apply preprocess function to every sample
tokenized_data = data["train"].map(preprocess, batched=True)

# add token pos, [CLS]
def add_token_positions(sample):
    offset = sample["offset_mapping"]
    input_ids = sample["input_ids"]
    cls_index = input_ids.index(tokenizer.cls_token_id)

    answer_start = sample["answers"]["answer_start"][0]
    answer_text = sample["answers"]["text"][0]
    answer_end = answer_start + len(answer_text)

    start_pos = end_pos = cls_index  # default fallback

    for i, (start, end) in enumerate(offset):
        if start <= answer_start < end:
            start_pos = i
        if start < answer_end <= end:
            end_pos = i
            break

    sample["start_positions"] = start_pos
    sample["end_positions"] = end_pos
    return sample

tokenized_data = tokenized_data.map(add_token_positions, remove_columns=["offset_mapping", "answers", "context", "question", "id"])

# print(tokenized_data)

save_dir = fine_tuned_model_path() / "deberta-v3-base-fine-tuned"

# training arguments, epochs, lr, logging loss
training_args = TrainingArguments(
    output_dir=save_dir,
    per_device_train_batch_size=2,
    num_train_epochs=4,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=1,
    save_steps=50,
    report_to="none"
)

data_collator = DefaultDataCollator()

# trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    tokenizer=tokenizer,
    data_collator=data_collator 
)

trainer.train()

loss_values = []
steps = []

for log in trainer.state.log_history:
    if "loss" in log:
        loss_values.append(log["loss"])
        steps.append(log["step"])

plt.figure()
plt.title(f'Deberta Fine Tune Loss Curves')
plt.plot(steps, loss_values)
plt.xlabel('steps')
plt.ylabel("Loss")
plt.legend(labels=['Loss'])
plt.show()

save_ft_m_dir = save_dir
save_ft_t_dir = f"{save_dir}-token"

# saved fine tuned model, alongside it's tokenizer
model.save_pretrained(save_ft_m_dir)
tokenizer.save_pretrained(save_ft_t_dir)