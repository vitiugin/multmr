from transformers import AutoTokenizer
from datasets import load_dataset, ClassLabel, Sequence

import os
import sys
from transformers import TrainingArguments, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss

from datasets import load_metric
import numpy as np
import evaluate

from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.3, beta=0.3, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

class MultiDistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model_1=None, teacher_model_2=None, teacher_model_3=None,**kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_1 = teacher_model_1
        self.teacher_2 = teacher_model_2
        self.teacher_3 = teacher_model_3
        # place teacher on same device as student
        self._move_model_to_device(self.teacher_1,self.model.device)
        self.teacher_1.eval()
        self._move_model_to_device(self.teacher_2,self.model.device)
        self.teacher_2.eval()
        self._move_model_to_device(self.teacher_3,self.model.device)
        self.teacher_3.eval()

    def compute_loss(self, model, inputs, return_outputs=False):

        # compute student output
        outputs_student = model(**inputs,output_hidden_states=True)
        student_loss=outputs_student.loss

        # compute teacher output
        with torch.no_grad():
          outputs_teacher_1 = self.teacher_1(**inputs,output_hidden_states=True)
          outputs_teacher_2 = self.teacher_2(**inputs,output_hidden_states=True)
          outputs_teacher_3 = self.teacher_3(**inputs,output_hidden_states=True)

        hidden_loss = []
        for num in range(1, len(outputs_student.hidden_states), 4):
            outputs_teacher = torch.stack([outputs_teacher_1.hidden_states[num] * 0.4,
                                          outputs_teacher_3.hidden_states[num] * 0.3,
                                          outputs_teacher_3.hidden_states[num] * 0.3
                                           ]).mean(dim=0)

            loss_logits = (nn.MSELoss()(
                F.log_softmax(outputs_student.hidden_states[num], dim=-1),
                F.softmax(outputs_teacher, dim=-1)))
            hidden_loss.append(loss_logits)
            
        hidden_loss = torch.mean(F.normalize(torch.stack(hidden_loss), dim=0))

        # let's test this approach with mean
        teachers_logits = torch.stack([outputs_teacher_1.logits  * 0.4, 
                                       outputs_teacher_2.logits * 0.3, 
                                       outputs_teacher_3.logits * 0.3
                                       ]).mean(dim=0)

        loss_logits = (CrossEntropyLoss()(
            F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
            F.softmax(teachers_logits / self.args.temperature, dim=-1)) * (self.args.temperature ** 2))


        loss = self.args.alpha * student_loss + (1. - self.args.alpha) * loss_logits + self.args.beta * student_loss + (1 - self.args.beta) * hidden_loss
        
        return (loss, outputs_student) if return_outputs else loss

# define metrics and metrics function
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
auc_metric = evaluate.load("roc_auc")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions[0], axis=1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    auc = auc_metric.compute(prediction_scores=predictions, references=labels)
    return {
        "accuracy": acc["accuracy"],
        "f1": f1["f1"],
        "auc": auc["roc_auc"]
    }

def process(examples):
    tokenized_inputs = tokenizer(
        examples["text"], padding=True, truncation=True, max_length=512
    )
    return tokenized_inputs


student_id = sys.argv[1]
teacher_id_1 = sys.argv[1]
teacher_id_2 = sys.argv[2]
teacher_id_3 = sys.argv[3]

# init tokenizer
teacher_tokenizer_1 = AutoTokenizer.from_pretrained(teacher_id_1)
teacher_tokenizer_2 = AutoTokenizer.from_pretrained(teacher_id_2)
teacher_tokenizer_3 = AutoTokenizer.from_pretrained(teacher_id_3)
student_tokenizer = AutoTokenizer.from_pretrained(student_id)

# init tokenizer

tokenizer = AutoTokenizer.from_pretrained(teacher_id_1)

dataset = load_dataset("data")
dataset = dataset.shuffle(seed=5)
tokenized_datasets = dataset.map(process, batched=True)
tokenized_datasets = tokenized_datasets.class_encode_column("label")
tokenized_datasets = tokenized_datasets.rename_column("label","labels")
tokenized_train = tokenized_datasets['train']
tokenized_val = tokenized_datasets['validation']
tokenized_test = tokenized_datasets['test']


# create label2id, id2label dicts for nice outputs for the model
labels = tokenized_train.features["labels"].names
num_labels = len(labels)
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label



# define training args
training_args = DistillationTrainingArguments(
    output_dir='logs',
    num_train_epochs=10,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64, 
    fp16=True,
    learning_rate=6e-5,
    seed=33,
    # logging & evaluation strategies
    logging_strategy="epoch", # to get more information to TB
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="auc",
    report_to="tensorboard",
    push_to_hub=False,
    hub_strategy="every_save",
    alpha=0.6,
    beta=0.5,
    temperature=2.0
    )

# define data_collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# define model
teacher_model_1 = AutoModelForSequenceClassification.from_pretrained(
    teacher_id_1,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

teacher_model_2 = AutoModelForSequenceClassification.from_pretrained(
    teacher_id_2,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

teacher_model_3 = AutoModelForSequenceClassification.from_pretrained(
    teacher_id_3,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)


# define student model
student_model = AutoModelForSequenceClassification.from_pretrained(
    student_id,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)


trainer = MultiDistillationTrainer(
    student_model,
    training_args,
    teacher_model_1=teacher_model_1,
    teacher_model_2=teacher_model_2,
    teacher_model_3=teacher_model_3,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.predict(tokenized_train).label_ids

trainer.save_model("model/distilled_model")