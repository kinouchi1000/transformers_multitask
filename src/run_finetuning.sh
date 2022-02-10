#! /bin/bash

. ./finetuning_constants.sh

python ./run_glue_edit.py \
    --model_name_or_path=${MODEL_NAME} \
    --do_train \
    --do_eval \
    --max_seq_length=${MAX_SEQ_LENGTH} \
    --per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --use_fast_tokenizer \
    --learning_rate=${LEARNING_RATE} \
    --num_train_epochs=${NUM_TRAIN_EPOCHS} \
    --output_dir=${OUTPUT_DIR} \
    --train_file_task1=../data/negaposi/train.csv \
    --train_file_task2=../data/category/train.csv \
    --validation_file_task1=../data/negaposi/valid.csv \
    --validation_file_task2=../data/category/valid.csv \
    --run_name ${PROJECT_NAME} \
    --logging_dir ${LOG_DIR} \
    --logging_strategy steps \
    --logging_steps ${STEPS} \
    --evaluation_strategy steps \
    --load_best_model_at_end\
    --eval_steps ${STEPS} \
    --metric_for_best_model eval_loss 
    
#     --overwrite_output_dir \