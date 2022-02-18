#! /bin/bash

set -eu

# BERT
# PROJECT_NAME=multitasking-bert-base-japanese-whole-word-masking-data2
# MODEL_NAME=cl-tohoku/bert-base-japanese-whole-word-masking
# . ./run_finetuning.sh

# PROJECT_NAME=multitasking-bert-base-japanese-v2
# MODEL_NAME=cl-tohoku/bert-base-japanese-v2
# . ./run_finetuning.sh


# RoBERTa
# PROJECT_NAME=multitasking-rinnna-japanese-roberta-base
# MODEL_NAME=rinna/japanese-roberta-base
# . ./run_finetuning.sh

# PROJECT_NAME=multitasking-waseda-japanese-roberta-base
# MODEL_NAME=nlp-waseda/roberta-base-japanese
# . ./run_finetuning.sh

# Q&A
# PROJECT_NAME=qa-rinna-roberta2
# MODEL_NAME=rinna/japanese-roberta-base
# TRAIN_PATH_TASK1=../data/negaposi/trainQA.csv
# TRAIN_PATH_TASK2=../data/category/trainQA.csv
# EVAL_PATH_TASK1=../data/negaposi/validQA.csv
# EVAL_PATH_TASK2=../data/category/validQA.csv
# . ./run_finetuning.sh

# data JED
# PROJECT_NAME=JED-rinna-roberta
# MODEL_NAME=rinna/japanese-roberta-base
# TRAIN_PATH_TASK1=../data_JED/negaposi/train.csv
# TRAIN_PATH_TASK2=../data_JED/category/train.csv
# EVAL_PATH_TASK1=../data_JED/negaposi/valid.csv
# EVAL_PATH_TASK2=../data_JED/category/valid.csv
# . ./run_finetuning.sh

# fix loss
# PROJECT_NAME=qa-rinna-roberta-fixLoss
# MODEL_NAME=rinna/japanese-roberta-base
# TRAIN_PATH_TASK1=../data/negaposi/trainQA.csv
# TRAIN_PATH_TASK2=../data/category/trainQA.csv
# EVAL_PATH_TASK1=../data/negaposi/validQA.csv
# EVAL_PATH_TASK2=../data/category/validQA.csv
# . ./run_finetuning.sh

# label smoothing
PROJECT_NAME=qa-rinna-roberta-label-smoothing005
MODEL_NAME=rinna/japanese-roberta-base
TRAIN_PATH_TASK1=../data/negaposi/trainQA.csv
TRAIN_PATH_TASK2=../data/category/trainQA.csv
EVAL_PATH_TASK1=../data/negaposi/validQA.csv
EVAL_PATH_TASK2=../data/category/validQA.csv
. ./run_finetuning.sh

