# MulTMR
Multiple Teachers Model for Ranking

The repository contains code implementation and dataset for paper **Multilingual Serviceability Model for Ranking Help Requests on Social Media during Crisis Events**.

## Data preparation

1. Download tweets based on ids' presented in the folder `data`.
2. Create 3 files: *train.csv*, *val.csv*, *test.csv*. Each file should contain two colimns: 'text' and 'label'.

Datasets for behavioral fine-tuning:
- sarcasm and irony detection dataset [link](https://www.kaggle.com/datasets/nikhiljohnk/tweets-with-sarcasm-and-irony)
- wuestion type classification dataset [link](https://www.kaggle.com/datasets/ananthu017/question-classification)

After downloading datasets, create 3 files *train.csv*, *val.csv*, *test.csv* and save the in the folder inside the `data` folder. Each file should contain two colimns: 'text' and 'label'.

## Installation

The code was tested on Python 3.10.

Install necessary dependencies with use of pip:
```
pip -r requirements.txt
```

## Model fine-tuning

1. Command for task-related fine-tuning:
```
python finetune.py bert data teacher_1
```

where
- bert -- name of the model (could be **bert** or **roberta**)
- data -- folder with data
- teacher_1 -- name of folder for saving the task-related Teacher model

2. Command for behavioral fine-tuning:
```
python finetune.py bert data/sarcasm teacher_2
```
where
- bert -- name of the model (could be **bert** or **roberta**)
- data/sarcasm -- folder with data for behavioral fine-tuning
- teacher_2 -- name of folder for saving the behavior-guided Teacher model

## Model distillation

Command for model distillation:

```
python distillation.py model/teacher_1 model/teacher_2 model/teacher_3
```
where `model/teacher_1`, `model/teacher_2`, and `model/teacher_3` - paths to finetuned models.