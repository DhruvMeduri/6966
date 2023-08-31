from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import os
import argparse
import random
import json

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, help='Directory where model checkpoints will be saved')
args = parser.parse_args()


dataset = load_dataset("imdb")

model_checkpoint = "microsoft/deberta-v3-large" 

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)


# Everything seems correct so far, need to fine tune the model. Do it later


num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

metric = load_metric('accuracy')

args = TrainingArguments(
    os.path.join(args.output_dir, "deberta-v3-large"),
    evaluation_strategy = "steps",
    save_strategy = "steps",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=5,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model= 'accuracy',
    eval_steps = 1000,
    save_steps = 1000
)

def compute_metrics(eval_pred):

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# To check the accuracy before training
print("To check the accuracy before training\n")
predictions = trainer.predict(tokenized_datasets["test"])
preds = np.argmax(predictions.predictions, axis=1)
print(metric.compute(predictions=preds, references=predictions.label_ids))

trainer.train()

print("To check the accuracy after training\n")
predictions = trainer.predict(tokenized_datasets["test"])
preds = np.argmax(predictions.predictions, axis=1)
print(metric.compute(predictions=preds, references=predictions.label_ids))

failures = []

for i in range(len(preds)):
    
    if preds[i] != predictions.label_ids[i]:
        failures.append(i)



random.shuffle(failures)
final_list = []

for i in range(10):

    ele = {"Review":tokenized_datasets["test"]["text"][failures[i]],"label":int(predictions.label_ids[failures[i]]),"predicted":int(preds[failures[i]])}
    final_list.append(ele)
    
with open("output.json", "w") as outfile:
    
    json.dump(final_list, outfile)
