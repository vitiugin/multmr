import os
import sys
import math

import evaluate
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, RobertaConfig
from transformers import AutoModelForSequenceClassification, AutoModelForPreTraining
from transformers import TrainingArguments, Trainer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# arguments
pretrained_model_name = sys.argv[1]
data_folder = sys.argv[1]
finetuned_model_name = sys.argv[3]

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    auc = auc_metric.compute(prediction_scores=predictions, references=labels)
    return {
        "accuracy": acc["accuracy"],
        "f1": f1["f1"],
        "auc": auc["roc_auc"]
    }

# pretrained model loading
if pretrained_model_name == 'bert':
    PRETRAINED_MODEL="bert-base-multilingual-cased" # 12 layers
    FROZEN_LAYERS=5
elif pretrained_model_name == 'roberta':
    PRETRAINED_MODEL="xlm-roberta-large" # 24 layers
    FROZEN_LAYERS=20
else:
    print('The script supports "bert" or "roberta" models only.')
 
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding=False, truncation=True)

def get_random_seed():
    return int.from_bytes(os.urandom(4), "big")


# dataset loading and splitting
dataset = load_dataset(data_folder) 
dataset = dataset.shuffle(5)
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.class_encode_column("label")
tokenized_datasets = tokenized_datasets.rename_column("label","labels")
tokenized_train = tokenized_datasets['train']
tokenized_val = tokenized_datasets['validation']
tokenized_test = tokenized_datasets['test']


# define metrics and metrics function
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
auc_metric = evaluate.load("roc_auc")


args_dict = {
        "evaluation_strategy": "steps",
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "learning_rate": 5e-5,
        "num_train_epochs": 10,
        "logging_first_step": True,
        "save_total_limit": 1,
        "fp16": True,
        "dataloader_num_workers": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "auc",
        "seed": get_random_seed(),
    }

freeze_layer_count = FROZEN_LAYERS

if freeze_layer_count:
    # We freeze here the embeddings of the model
    if pretrained_model_name == 'bert':
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False
    else:
        for param in model.roberta.embeddings.parameters():
            param.requires_grad = False
        
    if freeze_layer_count != -1:
	    # if freeze_layer_count == -1, we only freeze the embedding layer
	    # otherwise we freeze the first `freeze_layer_count` encoder layers
        if pretrained_model_name == 'bert':
            for layer in model.bert.encoder.layer[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False
        else:
            for layer in model.roberta.encoder.layer[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False
    
epoch_steps = len(tokenized_train) / args_dict["per_device_train_batch_size"]
args_dict["warmup_steps"] = math.ceil(epoch_steps)  # 1 epoch
args_dict["logging_steps"] = max(1, math.ceil(epoch_steps))  # 0.5 epoch
args_dict["save_steps"] = args_dict["logging_steps"]


training_args = TrainingArguments(output_dir="test_trainer", **args_dict)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    )

trainer.train()

trainer.predict(tokenized_test).metrics

trainer.save_model("model/" + finetuned_model_name)