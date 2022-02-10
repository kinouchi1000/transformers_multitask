#! /bin/bash

PROJECT_NAME=multitasking-bert-base-japanese-whole-word-masking-data
MODEL_NAME=cl-tohoku/bert-base-japanese-whole-word-masking
. ./run_finetuning.sh

# PROJECT_NAME=bert-base-japanese-whole-word-masking-data2
# MODEL_NAME=cl-tohoku/bert-base-japanese-whole-word-masking
# TRAIN_FILE=../data/trainData2/train.csv
# VALIDATION_FILE=../data/trainData2/eval.csv 
# . ./run_finetuning.sh