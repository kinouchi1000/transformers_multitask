#! /bin/bash

PROJECT_NAME=multitasking-bert-base-japanese-whole-word-masking-data2
MODEL_NAME=cl-tohoku/bert-base-japanese-whole-word-masking
. ./run_finetuning.sh

PROJECT_NAME=multitasking-bert-base-japanese-v2
MODEL_NAME=cl-tohoku/bert-base-japanese-v2
. ./run_finetuning.sh